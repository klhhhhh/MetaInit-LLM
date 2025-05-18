import torch
import torch.nn.functional as F

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from omegaconf import OmegaConf
from .model_loader import load_model_to_cpu
from .model_builder import build_model

class ModelProjectionUtils:
    def __init__(self, small_model_path, large_model_cfg_path, device="cpu"):
        self.device = device
        self.small_model = self._load_small_model(small_model_path)
        self.large_model, self.large_state_dict = self._load_large_model(large_model_cfg_path)

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

    def project_parameters(self, rank=64):
        small_state_dict = self.small_model.state_dict()
        large_num_layers = self.large_model.cfg.num_layers  # Assumes large_model has a cfg attribute

        # Step 1: Interpolate parameters between layers
        if hasattr(self.small_model.decoder, "layers") and hasattr(self.large_model.decoder, "layers"):
            small_layers = self.small_model.decoder.layers
            interpolated_params_list = self.expand_layers(small_layers, large_num_layers)

            # Step 2: Map interpolated parameters to large model layers and perform hidden_size projection
            for layer_idx, large_layer in enumerate(self.large_model.decoder.layers):
                interpolated_params = interpolated_params_list[layer_idx]
                
                for name, param_large in large_layer.named_parameters():
                    if name not in interpolated_params:
                        continue
                    param_small = interpolated_params[name]

                    if param_small.shape == param_large.shape:
                        param_large.data.copy_(param_small)
                    elif len(param_small.shape) == 2:
                        # Linear weights: hidden_size mismatch
                        projected_param = self.low_rank_projection(
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
                self.large_state_dict[name] = self.low_rank_projection(
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