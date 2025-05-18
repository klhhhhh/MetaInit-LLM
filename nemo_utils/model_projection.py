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
            model = load_model_to_cpu(small_model_path)
            return model

    def _load_large_model(self, cfg_name):
        large_model = build_model(cfg_name)
        return large_model, large_model.state_dict()

    def layer_interpolation(self, small_layers, target_num_layers):
        small_num_layers = len(small_layers)
        mapped_layers = []
        for i in range(target_num_layers):
            pos = i * (small_num_layers - 1) / (target_num_layers - 1)
            low_idx = int(pos)
            high_idx = min(low_idx + 1, small_num_layers - 1)
            alpha = pos - low_idx
            interpolated = (1 - alpha) * small_layers[low_idx] + alpha * small_layers[high_idx]
            mapped_layers.append(interpolated)
        return mapped_layers

    def low_rank_projection(self, W_small, target_shape, rank=64):
        out_large, in_large = target_shape
        in_small = W_small.shape[1]

        # Random projection matrices (can be learned later)
        P_in = torch.randn(in_large, in_small, device=W_small.device)
        P_out = torch.randn(out_large, W_small.shape[0], device=W_small.device)

        W_large = P_out @ W_small @ P_in.T
        return W_large

    def project_parameters(self, rank=64):
        small_state_dict = self.small_model.state_dict()

        for name, param_small in small_state_dict.items():
            if name not in self.large_state_dict:
                continue

            param_large = self.large_state_dict[name]
            if param_small.shape == param_large.shape:
                self.large_state_dict[name] = param_small  # Directly copy

            elif len(param_small.shape) == 2:
                # Linear layers: hidden size mismatch
                self.large_state_dict[name] = self.low_rank_projection(
                    param_small, param_large.shape, rank=rank
                )

            elif len(param_small.shape) == 1:
                # Bias / LayerNorm params: simple interpolation
                new_shape = (param_large.shape[0],)
                interpolated = F.interpolate(
                    param_small.unsqueeze(0).unsqueeze(0),
                    size=new_shape,
                    mode="linear",
                    align_corners=True
                ).squeeze()
                self.large_state_dict[name] = interpolated

    def save_projected_model(self, save_path):
        torch.save(self.large_state_dict, save_path)
        print(f"Projected model weights saved to {save_path}")


# =====================
# ✅ Usage Example:
# =====================

if __name__ == "__main__":
    projector = ModelProjectionUtils(
        small_model_path="/work/hdd/bdrw/klin4/checkpoints/nemo/gpt/megatron_gpt.nemo", 
        large_model_cfg_path=""
    )

    projector.project_parameters(rank=64)
    projector.save_projected_model("checkpoints/large_model_projected_init.pt")
