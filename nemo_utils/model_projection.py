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

    def expand_layers(self, small_layers, target_num_layers):
        """
        Use inter-layer interpolation to expand the number of layers 
        from the small model to match the large model.
        The parameters of each layer are calculated using linear interpolation.
        """
        small_num_layers = len(small_layers)
        expanded_layers = []

        for i in range(target_num_layers):
            pos = i * (small_num_layers - 1) / (target_num_layers - 1)
            low_idx = int(pos)
            high_idx = min(low_idx + 1, small_num_layers - 1)
            alpha = pos - low_idx

            # Interpolate parameters layer by layer
            interpolated_layer = {}
            for (param_name, param_low), (_, param_high) in zip(
                small_layers[low_idx].named_parameters(),
                small_layers[high_idx].named_parameters()
            ):
                interpolated_param = (1 - alpha) * param_low.data + alpha * param_high.data
                interpolated_layer[param_name] = interpolated_param

            expanded_layers.append(interpolated_layer)

        return expanded_layers

    def low_rank_projection(self, W_small, target_shape, rank=64):
        """
        Low-rank projection to adjust hidden_size.
        """
        out_large, in_large = target_shape
        out_small, in_small = W_small.shape

        # If already aligned, return directly
        if (out_large, in_large) == (out_small, in_small):
            return W_small

        # Generate random low-rank projection matrices
        P_in = torch.randn(in_small, in_large, device=W_small.device) / (in_small ** 0.5)
        P_out = torch.randn(out_large, out_small, device=W_small.device) / (out_small ** 0.5)

        W_large = P_out @ W_small @ P_in  # Shape: (out_large, in_large)
        return W_large

    def project_parameters(self, rank=64):
        small_state_dict = self.small_model.state_dict()

        for name, param_small in small_state_dict.items():
            if name not in self.large_state_dict:
                continue

            param_large = self.large_state_dict[name]

            if param_small.shape == param_large.shape:
                self.large_state_dict[name] = param_small  # Direct copy

            elif len(param_small.shape) == 2:
                # Linear layer weights projection
                self.large_state_dict[name] = self.low_rank_projection(
                    param_small, param_large.shape, rank=rank
                )

            elif len(param_small.shape) == 1:
                # Bias or LayerNorm weights interpolation
                new_size = param_large.shape[0]
                old_size = param_small.shape[0]

                if new_size == old_size:
                    self.large_state_dict[name] = param_small
                else:
                    interpolated = F.interpolate(
                        param_small.unsqueeze(0).unsqueeze(0),  # Add dummy batch and channel dims
                        size=new_size,
                        mode="linear",
                        align_corners=True
                    ).squeeze()
                    self.large_state_dict[name] = interpolated

    def save_projected_model(self, save_path):
        os.makedirs(Path(save_path).parent, exist_ok=True)
        torch.save(self.large_state_dict, save_path)
        print(f"✅ Projected model weights saved to: {save_path}")

# =====================
# ✅ Usage Example:
# =====================

if __name__ == "__main__":
    projector = ModelProjectionUtils(
        small_model_path="/work/hdd/bdrw/klin4/checkpoints/nemo/gpt/megatron_gpt.nemo", 
        large_model_cfg_path="megatron_gpt_350m_config"
    )

    projector.project_parameters(rank=64)
    projector.save_projected_model("checkpoints/large_model_projected_init.pt")
