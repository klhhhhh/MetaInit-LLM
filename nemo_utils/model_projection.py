import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from omegaconf import OmegaConf
from model_loader import load_model_to_cpu
from model_builder import build_model

class LoRAProjector(nn.Module):
    def __init__(self, d_small, d_large, rank=64):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d_large, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, d_small) * 0.01)

    def forward(self, W_small):
        P = self.A @ self.B  # [d_large, d_small]
        return P @ W_small @ P.T  # [d_large, d_large]

class ModelProjectionUtils:
    def __init__(self, small_model_path, large_model_cfg_path, device="cpu"):
        self.device = device
        self.small_model = self._load_small_model(small_model_path)
        self.large_model, self.large_state_dict = self._load_large_model(large_model_cfg_path)
        self.lora_modules = {}  # name → LoRA projector

    def _load_small_model(self, path):
        if self.device == "cpu":
            model = load_model_to_cpu(path)
        else:
            model = torch.load(path, map_location=self.device)
        return model

    def _load_large_model(self, cfg_name):
        large_model = build_model(cfg_name)
        return large_model, large_model.state_dict()

    def expand_layers(self, small_layers, target_num_layers):
        """
        Compute interpolated parameters of small model layers to map to large model layers.
        Returns a list of interpolated parameter dictionaries, without directly constructing Layer objects.
        """
        small_num_layers = len(small_layers)
        expanded_params = []

        for i in range(target_num_layers):
            pos = i * (small_num_layers - 1) / (target_num_layers - 1)
            low_idx = int(pos)
            high_idx = min(low_idx + 1, small_num_layers - 1)
            alpha = pos - low_idx

            interpolated_layer_params = {}
            for (name, param_low), (_, param_high) in zip(
                small_layers[low_idx].named_parameters(), 
                small_layers[high_idx].named_parameters()
            ):
                interpolated_param = (1 - alpha) * param_low.data + alpha * param_high.data
                interpolated_layer_params[name] = interpolated_param

            expanded_params.append(interpolated_layer_params)

        return expanded_params


    def _interpolate_layer(self, layer_low, layer_high, alpha):
        """
        Interpolate all parameters of a single Layer and return a new Layer object.
        """
        new_layer = type(layer_low)()  # Assumes the layer class can be constructed without arguments
        for (name, param_low), (_, param_high) in zip(
            layer_low.named_parameters(), layer_high.named_parameters()
        ):
            interpolated_param = (1 - alpha) * param_low.data + alpha * param_high.data
            getattr(new_layer, name).data.copy_(interpolated_param)
        return new_layer

    def lora_style_projection(self, W_small, target_shape, rank=64):
        """
        Perform LoRA-style low-rank projection for initializing larger model weights.
        W_large = (A @ B) @ W_small @ (A @ B).T
        A: [d_large, r], B: [r, d_small]
        """
        d_large, d_small = target_shape

        # Initialize A and B
        A = torch.randn(d_large, rank, device=W_small.device) * 0.01
        B = torch.randn(rank, d_small, device=W_small.device) * 0.01

        P = A @ B  # [d_large, d_small]
        W_large = P @ W_small @ P.T  # final shape: [d_large, d_large]

        return W_large
    
    def low_rank_projection(self, W_small, target_shape, rank=64):
        """
        Perform low-rank projection on the hidden_size dimension.
        """
        out_large, in_large = target_shape
        out_small, in_small = W_small.shape

        if (out_large, in_large) == (out_small, in_small):
            return W_small

        P_in = torch.randn(in_small, in_large, device=W_small.device) / (in_small ** 0.5)
        P_out = torch.randn(out_large, out_small, device=W_small.device) / (out_small ** 0.5)

        W_large = P_out @ W_small @ P_in  # Shape: (out_large, in_large)
        return W_large

    def project_parameters_learnable(self, rank=64):
        small_state_dict = self.small_model.model.state_dict()
        large_num_layers = self.large_model.cfg.num_layers  # Assumes large_model has a cfg attribute

        # Step 1: Interpolate parameters between layers
        if hasattr(self.small_model.model.decoder, "layers") and hasattr(self.large_model.model.decoder, "layers"):
            small_layers = self.small_model.model.decoder.layers
            interpolated_params_list = self.expand_layers(small_layers, large_num_layers)

            # Step 2: Map interpolated parameters to large model layers and perform hidden_size projection
            for layer_idx, large_layer in enumerate(self.large_model.model.decoder.layers):
                interpolated_params = interpolated_params_list[layer_idx]

                for name, param_large in large_layer.named_parameters():
                    if name not in interpolated_params:
                        continue
                    param_small = interpolated_params[name]

                    if param_small.shape == param_large.shape:
                        param_large.data.copy_(param_small)
                    elif len(param_small.shape) == 2:
                        projector = LoRAProjector(param_small.shape[1], param_large.shape[0], rank)
                        projected_param = projector(param_small)
                        param_large.data.copy_(projected_param)

                        full_name = f"decoder.layers.{layer_idx}.{name}"
                        self.lora_modules[full_name] = projector
                        self.large_model.add_module(f"lora_proj_{full_name.replace('.', '_')}", projector)
                    elif len(param_small.shape) == 1:
                        # Bias / LayerNorm weights interpolation
                        new_size = param_large.shape[0]
                        old_size = param_small.shape[0]

                        if new_size == old_size:
                            param_large.data.copy_(param_small)
                        else:
                            interpolated = F.interpolate(
                                param_small.unsqueeze(0).unsqueeze(0),
                                size=new_size,
                                mode="linear",
                                align_corners=True
                            ).squeeze()
                            param_large.data.copy_(interpolated)

        # Step 3: Handle direct mapping of non-layer parameters (e.g., embedding and final layer norm)
        for name, param_small in small_state_dict.items():
            if 'layers' in name or name not in self.large_state_dict:
                continue

            param_large = self.large_state_dict[name]
            if param_small.shape == param_large.shape:
                self.large_state_dict[name] = param_small
            elif len(param_small.shape) == 2:
                projector = LoRAProjector(param_small.shape[1], param_large.shape[0], rank)
                projected_param = projector(param_small)
                self.large_state_dict[name] = projected_param

                self.lora_modules[name] = projector
                self.large_model.add_module(f"lora_proj_{name.replace('.', '_')}", projector)
            elif len(param_small.shape) == 1:
                new_size = param_large.shape[0]
                old_size = param_small.shape[0]
                if new_size == old_size:
                    self.large_state_dict[name] = param_small
                else:
                    interpolated = F.interpolate(
                        param_small.unsqueeze(0).unsqueeze(0),
                        size=new_size,
                        mode="linear",
                        align_corners=True
                    ).squeeze()
                    self.large_state_dict[name] = interpolated

    def project_parameters(self, rank=64):
        small_state_dict = self.small_model.model.state_dict()
        large_num_layers = self.large_model.cfg.num_layers  # Assumes large_model has a cfg attribute

        # Step 1: Interpolate parameters between layers
        if hasattr(self.small_model.model.decoder, "layers") and hasattr(self.large_model.model.decoder, "layers"):
            small_layers = self.small_model.model.decoder.layers
            interpolated_params_list = self.expand_layers(small_layers, large_num_layers)

            # Step 2: Map interpolated parameters to large model layers and perform hidden_size projection
            for layer_idx, large_layer in enumerate(self.large_model.model.decoder.layers):
                interpolated_params = interpolated_params_list[layer_idx]
                
                for name, param_large in large_layer.named_parameters():
                    if name not in interpolated_params:
                        continue
                    param_small = interpolated_params[name]

                    if param_small.shape == param_large.shape:
                        param_large.data.copy_(param_small)
                    elif len(param_small.shape) == 2:
                        # Linear weights: hidden_size mismatch
                        projected_param = self.lora_style_projection(
                            param_small, param_large.shape, rank=rank
                        )
                        param_large.data.copy_(projected_param)
                    elif len(param_small.shape) == 1:
                        # Bias / LayerNorm weights interpolation
                        new_size = param_large.shape[0]
                        old_size = param_small.shape[0]

                        if new_size == old_size:
                            param_large.data.copy_(param_small)
                        else:
                            interpolated = F.interpolate(
                                param_small.unsqueeze(0).unsqueeze(0),
                                size=new_size,
                                mode="linear",
                                align_corners=True
                            ).squeeze()
                            param_large.data.copy_(interpolated)

        # Step 3: Handle direct mapping of non-layer parameters (e.g., embedding and final layer norm)
        for name, param_small in small_state_dict.items():
            if 'layers' in name or name not in self.large_state_dict:
                continue

            param_large = self.large_state_dict[name]
            if param_small.shape == param_large.shape:
                self.large_state_dict[name] = param_small
            elif len(param_small.shape) == 2:
                self.large_state_dict[name] = self.lora_style_projection(
                    param_small, param_large.shape, rank=rank
                )
            elif len(param_small.shape) == 1:
                new_size = param_large.shape[0]
                old_size = param_small.shape[0]
                if new_size == old_size:
                    self.large_state_dict[name] = param_small
                else:
                    interpolated = F.interpolate(
                        param_small.unsqueeze(0).unsqueeze(0),
                        size=new_size,
                        mode="linear",
                        align_corners=True
                    ).squeeze()
                    self.large_state_dict[name] = interpolated


    def save_projected_model(self, save_path):
        os.makedirs(Path(save_path).parent, exist_ok=True)
        torch.save(self.large_state_dict, save_path)
        print(f"✅ Projected model weights saved to: {save_path}")


if __name__ == "__main__":
    # import os
    # import argparse

    # parser = argparse.ArgumentParser(description="Project small model weights to large model.")
    # parser.add_argument("--small_model_path", type=str, required=True, help="Path to the small model checkpoint.")
    # parser.add_argument("--large_model_cfg_path", type=str, required=True, help="Path to the large model config.")
    # parser.add_argument("--save_path", type=str, required=True, help="Path to save the projected model weights.")
    # parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to load the model.")

    # args = parser.parse_args()

    small_model_path = "/work/hdd/bdrw/klin4/checkpoints/nemo/gpt/megatron_gpt.nemo"
    large_model_cfg_name = "megatron_gpt_350m_config"
    device = "cpu"
    utils = ModelProjectionUtils(small_model_path, large_model_cfg_name, device)
    utils.project_parameters()
    # utils.save_projected_model(args.save_path)