import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from omegaconf import OmegaConf
from model_loader import load_model_to_cpu
from model_builder import build_model

class ProjectedLinear(nn.Module):
    def __init__(self, W_small: torch.Tensor, d_large: int, rank: int = 64):
        super().__init__()
        d_small_out, d_small_in = W_small.shape

        # register buffer not parameter, save memory
        self.register_buffer('W_small', W_small.clone().detach())  # not a Parameter

        self.A = nn.Parameter(torch.randn(d_large, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, d_small_out) * 0.01)

    def forward(self, x):
        P = self.A @ self.B
        W_large = P @ self.W_small @ P.T
        return F.linear(x, W_large)

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

    def dispatch_projection(self, name, W_small, target_shape, projection_rank=32):
        """
        Determine the projection method based on the parameter name:
        - qkv: Split into Q, K, V and use symmetric projection
        - linear_proj: Use symmetric projection
        - mlp (fc1, fc2): Use asymmetric projection
        """
        if "linear_qkv.weight" in name and W_small.shape[0] % 3 == 0:
            # Split into Q, K, V
            qkv_chunks = torch.chunk(W_small, 3, dim=0)
            out_chunks = target_shape[0] // 3
            W_large_chunks = []

            for i, qkv in enumerate(qkv_chunks):
                projected = self.lora_style_projection_symmetric(
                    qkv, (out_chunks, target_shape[1]), projection_rank
                )
                W_large_chunks.append(projected)

            return torch.cat(W_large_chunks, dim=0)

        elif "linear_proj.weight" in name:
            return self.lora_style_projection_symmetric(W_small, target_shape, projection_rank)

        elif "mlp" in name or "fc1" in name or "fc2" in name:
            return self.lora_style_projection_asymmetric(W_small, target_shape, projection_rank)

        else:
            # Default to symmetric projection (you can also raise a warning)
            return self.lora_style_projection_symmetric(W_small, target_shape, projection_rank)

    
    def lora_style_projection_symmetric(self, W_small, target_shape, rank=32):
        """
        Symmetric LoRA projection: W_large = P @ W_small @ P^T
        Suitable for matrices like self_attention.linear_proj.weight with shape [h, h].
        """
        d_out, d_in = target_shape
        d_s_out, d_s_in = W_small.shape

        A = torch.randn(d_out, rank, device=W_small.device) * 0.01
        B = torch.randn(rank, d_s_out, device=W_small.device) * 0.01
        P = A @ B  # [d_out, d_s_out]

        W_large = P @ W_small @ P.T  # [d_out, d_out]
        return W_large

    def lora_style_projection_asymmetric(self, W_small, target_shape, projection_rank=32):
        """
        Perform LoRA-style low-rank projection with additional factorization (projection_rank) to reduce computation.
        
        W_large = (A1 @ A2) @ W_small @ (B1 @ B2)
        A1: [out_large, projection_rank]
        A2: [projection_rank, out_small]
        B1: [in_small, projection_rank]
        B2: [projection_rank, in_large]
        
        Final shape: [out_large, in_large]
        """
        out_large, in_large = target_shape
        out_small, in_small = W_small.shape

        # Low-rank factors for A and B
        A1 = torch.randn(out_large, projection_rank, device=W_small.device) * 0.01
        A2 = torch.randn(projection_rank, out_small, device=W_small.device) * 0.01
        B1 = torch.randn(in_small, projection_rank, device=W_small.device) * 0.01
        B2 = torch.randn(projection_rank, in_large, device=W_small.device) * 0.01

        # Final projection: W_large = (A1 @ A2) @ W_small @ (B1 @ B2)
        W_large = (A1 @ (A2 @ W_small @ B1)) @ B2  # shape: [out_large, in_large]

        return W_large

    def replace_with_projected_linear(self, parent_module, attr_name, W_small, out_dim, rank):
        projected = ProjectedLinear(W_small, out_dim, rank)
        setattr(parent_module, attr_name, projected)
        print(f"[Learnable] Replaced {attr_name} with ProjectedLinear")

    def project_parameters(self, rank=64, learnable=False):
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
                    print(name)
                    param_small = interpolated_params[name]

                    if param_small.shape == param_large.shape:
                        param_large.data.copy_(param_small)
                    elif len(param_small.shape) == 2:
                        if learnable:
                            module_path = name.split(".")[0]
                            self.replace_with_projected_linear(
                                getattr(large_layer, module_path),
                                name.split(".")[1],
                                param_small,
                                param_large.shape[0],
                                rank
                            )
                        else:
                            projected_param = self.dispatch_projection(
                                name=name,
                                W_small=param_small,
                                target_shape=param_large.shape,
                                projection_rank=rank
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
                if learnable:
                    print(f"[Learnable] Non-layer {name} projection skipped (requires custom module override)")
                else:
                    projected_param = self.dispatch_projection(
                        name=name,
                        W_small=param_small,
                        target_shape=param_large.shape,
                        projection_rank=rank
                    )
                    self.large_state_dict[name] = projected_param
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