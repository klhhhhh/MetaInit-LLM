import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from omegaconf import OmegaConf

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

from projection.layer_projection import (
    ColumnParallelLinearWithProjector,
    RowParallelLinearWithProjector)

from projection.projection_init import svd_lora_init_from_small

from nemo_utils.model_loader import load_model_to_cpu
from nemo_utils.model_builder import build_model

class ModelProjectionUtils:
    def __init__(self, small_model_path, large_model_cfg_path, callbacks, device="cpu"):
        self.device = device
        self.small_model = self._load_small_model(small_model_path)
        self.large_model, self.large_state_dict, self.large_trainer, self.large_exp_manager, self.dtype, self.num_heads = self._load_large_model(large_model_cfg_path, callbacks)
        self._set_dtype(self.dtype)
        self.lora_modules = {}  # name → LoRA projector

    def _set_dtype(self, dtype):
        """ Set the dtype based on the precision string. """
        precision_str = str(dtype).lower()
        if precision_str in ["bf16", "bf16-mixed"]:
            self.dtype = torch.bfloat16
        elif precision_str in ["16", "fp16", "16-mixed"]:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

    def _load_small_model(self, path):
        if self.device == "cpu":
            model = load_model_to_cpu(path)
        else:
            model = torch.load(path, map_location=self.device)
        return model

    def _load_large_model(self, cfg_name, callbacks=None):
        large_model, trainer, exp_manager, dtype, num_heads = build_model(cfg_name, callbacks=callbacks)
        return large_model, large_model.state_dict(), trainer, exp_manager, dtype, num_heads
    
    def get_large_model_trainer(self):
        """
        Returns the trainer associated with the large model.
        This is useful for training or evaluation after projection.
        """
        return self.large_trainer

    def get_large_model_exp_manager(self):
        """
        Returns the experiment manager associated with the large model.
        This is useful for managing checkpoints, logs, etc.
        """
        return self.large_exp_manager

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

    # ======== Renamed: Frobenius-norm matching (kept to avoid collision with row-std version) ========
    def normalize_projection_fro(self, W_projected, W_target):
        """
        Normalize the projected weight matrix to match the Frobenius norm of the target matrix.
        """
        projected_norm = torch.norm(W_projected, p='fro')
        target_norm = torch.norm(W_target, p='fro')
        if projected_norm > 0:
            print(f"Normalizing projection (Fro) from {projected_norm:.4f} to {target_norm:.4f}")
            W_projected = W_projected * (target_norm / projected_norm)
        return W_projected
    
    # ======== Renamed: row-wise std alignment (default calibration) ========
    def normalize_projection_rowstd(self, W_large: torch.Tensor, target_param: torch.Tensor,
                                    nonlin: str = 'gelu', use_target_stats: bool = True,
                                    eps: float = 1e-8) -> torch.Tensor:
        """
        One-time calibration of W_large via row-wise std alignment:
        - Prefer aligning to the row-wise std of target_param (stable).
        - Fallback to theoretical fan-in std (Xavier/He) if no target stats are available.
        """
        W = W_large
        if use_target_stats and target_param is not None and target_param.numel() > 0 \
        and target_param.shape == W.shape:
            tgt = target_param.std(dim=1, keepdim=True).clamp_min(eps)  # [out,1]
        else:
            # Fan-in target: gain / sqrt(fan_in)
            fan_in = W.size(1)
            gain = (2.0 ** 0.5) if nonlin in ('relu', 'gelu') else 1.0
            tgt_val = gain / (fan_in ** 0.5)
            tgt = W.new_full((W.size(0), 1), tgt_val)

        cur = W.std(dim=1, keepdim=True).clamp_min(eps)
        scale = tgt / cur
        return W * scale  # Row-wise scaling

    def _orthogonal_factors(self, out_dim, in_dim_small, rank, device, spectral_scale=0.1):
        A = torch.empty(out_dim, rank, device=device)
        B = torch.empty(rank, in_dim_small, device=device)
        torch.nn.init.orthogonal_(A)        # Columns of A are approximately orthogonal
        torch.nn.init.orthogonal_(B.T)      # Rows of B are approximately orthogonal -> Columns of B are approximately orthogonal
        A.mul_(spectral_scale)
        B.mul_(spectral_scale)
        return A, B
    
    def _apply_qk_head_scaling(self, W_q_or_k: torch.Tensor, num_heads: int):
        d_head = W_q_or_k.size(0) // num_heads
        return W_q_or_k / (d_head ** 0.5)

    # ========= Minimal change wrapper: call imported SVD initializer and row-std calibrate =========
    def svd_small_angle_project(self, W_small: torch.Tensor,
                                target_shape: tuple,
                                target_param: torch.Tensor,
                                r: int = None,
                                theta_deg: float = 8.0) -> torch.Tensor:
        """
        Use svd_lora_init_from_small from projection_init.py to upscale W_small to (d_b_out, d_b_in),
        then apply row-wise std alignment against target_param.
        """
        d_b_out, d_b_in = target_shape
        _,_,_,_,ret = svd_lora_init_from_small(
            W_s=W_small,
            d_b_out=d_b_out,
            d_b_in=d_b_in,
            r=r,
            theta_deg=theta_deg,
            device=W_small.device,
            dtype=W_small.dtype,
            return_Wb0=True,
        )
        # The imported function returns (A_out, B_out, A_in, B_in, Wb0) when return_Wb0=True.
        # Keep it robust in case of signature differences.
        Wb0 = ret[-1] if isinstance(ret, (tuple, list)) and len(ret) >= 1 else ret
        Wb0 = self.normalize_projection_rowstd(Wb0, target_param)
        return Wb0

    def dispatch_projection(self, name, W_small, target_shape, target_param, projection_rank=32,
                            svd_rank=None, theta_deg: float = 8.0):
        """
        Decide projection method by parameter name (non-learnable path):
        - Prefer SVD + small-angle mixing (via svd_lora_init_from_small).
        - QKV: split and apply SVD per chunk.
        - Others: apply SVD directly.
        Keep projection_rank in the signature for backward compatibility (unused here).
        """
        self.current_param_name = name

        if "linear_qkv.weight" in name and W_small.shape[0] % 3 == 0:
            # Split into Q, K, V
            q, k, v = torch.chunk(W_small, 3, dim=0)
            out_chunks = target_shape[0] // 3
            tgt_q = target_param[:out_chunks, :target_shape[1]]
            tgt_k = target_param[out_chunks:2*out_chunks, :target_shape[1]]
            tgt_v = target_param[2*out_chunks:, :target_shape[1]]

            Wq = self.svd_small_angle_project(q, (out_chunks, target_shape[1]), tgt_q, r=svd_rank, theta_deg=theta_deg)
            Wk = self.svd_small_angle_project(k, (out_chunks, target_shape[1]), tgt_k, r=svd_rank, theta_deg=theta_deg)
            Wv = self.svd_small_angle_project(v, (out_chunks, target_shape[1]), tgt_v, r=svd_rank, theta_deg=theta_deg)

            # If you need extra 1/sqrt(d_head) scaling for Q/K, uncomment:
            # Wq = self._apply_qk_head_scaling(Wq, self.num_heads)
            # Wk = self._apply_qk_head_scaling(Wk, self.num_heads)

            return torch.cat([Wq, Wk, Wv], dim=0)

    # Default: also use SVD small-angle projection
        elif "linear_proj.weight" in name or "mlp" in name or "fc1" in name or "fc2" in name:
            return self.svd_small_angle_project(W_small, target_shape, target_param, r=svd_rank, theta_deg=theta_deg)

        else:
            return self.svd_small_angle_project(W_small, target_shape, target_param, r=svd_rank, theta_deg=theta_deg)
    
    def lora_style_projection_symmetric(self, W_small, target_shape, target_param, rank=32,
                                    spectral_scale=0.1, num_heads=None, is_q=False, is_k=False):
        out_large, in_large = target_shape
        assert out_large == in_large, "symmetric projection is only used for square matrices"

        A, B = self._orthogonal_factors(out_large, W_small.shape[0], rank, W_small.device, spectral_scale)
        P = A @ B                                 # [out_large, out_small]
        W_large = P @ W_small @ P.T               # [out_large, out_large]

        # Additional scaling for Q/K weights (if this is q or k weights)
        if num_heads is not None and (is_q or is_k):
            W_large = self._apply_qk_head_scaling(W_large, num_heads)

        # Use the row-wise std alignment
        W_large = self.normalize_projection_rowstd(W_large, target_param)
        return W_large


    def lora_style_projection_asymmetric(self, W_small, target_shape, target_param, rank=32, spectral_scale=0.1):
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

        A1, A2 = self._orthogonal_factors(out_large, out_small, rank, W_small.device, spectral_scale)
        B1, B2 = self._orthogonal_factors(in_small, in_large, rank, W_small.device, spectral_scale)
        W_large = (A1 @ (A2 @ W_small @ B1)) @ B2

        # Use the row-wise std alignment
        W_large = self.normalize_projection_rowstd(W_large, target_param)
        return W_large

    def _bind_init_kwargs_from_record(self, layer, rec: dict) -> dict:
        """
        Build a kwargs dict for re-instantiation from INIT_RECORDS entry:
        rec = {"class_name":..., "instance":..., "args": tuple, "kwargs": dict}
        Falls back to attributes on the live layer when needed.
        """
        args = rec.get("args", ()) or ()
        kwargs = rec.get("kwargs", {}) or {}

        # Try binding to the layer class __init__ signature (to map positional -> names)
        try:
            import inspect
            sig = inspect.signature(layer.__class__.__init__)
            ba = sig.bind_partial(layer, *args, **kwargs)  # include self
            bound = {k: v for k, v in ba.arguments.items() if k != "self"}
        except Exception:
            # If binding fails, at least use the recorded kwargs
            bound = dict(kwargs)

        # Fill missing essentials from the live object
        bound.setdefault("input_size", getattr(layer, "input_size", None))
        bound.setdefault("output_size", getattr(layer, "output_size", None))
        bound.setdefault("config", getattr(layer, "config", None))
        # Column vs Row specifics
        if isinstance(layer, ColumnParallelLinear):
            bound.setdefault("gather_output", getattr(layer, "gather_output", False))
            bound.setdefault("skip_bias_add", getattr(layer, "skip_bias_add", False))
            bound.setdefault("stride", 1)
            bound.setdefault("keep_master_weight_for_test", False)
            bound.setdefault("skip_weight_param_allocation", False)
            bound.setdefault("embedding_activation_buffer", getattr(layer, "embedding_activation_buffer", None))
            bound.setdefault("grad_output_buffer", getattr(layer, "grad_output_buffer", None))
            bound.setdefault("is_expert", getattr(layer, "is_expert", False))
            bound.setdefault("tp_comm_buffer_name", None)
            bound.setdefault("disable_grad_reduce", getattr(layer, "disable_grad_reduce", False))
        elif isinstance(layer, RowParallelLinear):
            bound.setdefault("input_is_parallel", getattr(layer, "input_is_parallel", False))
            bound.setdefault("skip_bias_add", getattr(layer, "skip_bias_add", False))
            bound.setdefault("stride", 1)
            bound.setdefault("keep_master_weight_for_test", False)
            bound.setdefault("is_expert", getattr(layer, "is_expert", False))
            bound.setdefault("tp_comm_buffer_name", None)

        # bias flag: original constructor expects a bool
        if "bias" not in bound:
            bound["bias"] = getattr(layer, "bias", None) is not None

        # init_method: if not recorded, provide a sane default
        if "init_method" not in bound or bound["init_method"] is None:
            bound["init_method"] = torch.nn.init.xavier_uniform_

        return bound


    def _filter_kwargs_for_ctor(self, cls, kwargs: dict) -> dict:
        """Keep only kwargs accepted by cls.__init__, drop the rest."""

        import inspect

        sig = inspect.signature(cls.__init__)
        allowed = {p.name for p in sig.parameters.values()}
        allowed.discard("self")
        return {k: v for k, v in kwargs.items() if k in allowed}

    def replace_layer_with_projector(
        self,
        parent_module,
        attr_name,
        W_small: torch.Tensor,
        target_shape,          # (out, in) — can be used for sanity check
        target_dtype,          # could be torch.bfloat16 / torch.float16 / torch.float32
        rank: int,
        name: str,
    ):
        from nemo_utils.init_hook import INIT_RECORDS

        layer = getattr(parent_module, attr_name)

        # decide projection type
        if "linear_qkv" in name:
            projection_type = "qkv"
        elif "linear_proj" in name:
            projection_type = "symmetric"
        elif "mlp" in name or "fc1" in name or "fc2" in name:
            projection_type = "asymmetric"
        else:
            raise ValueError(f"Unknown layer for projector replacement: {name}")

        # fetch the record and rebuild kwargs
        rec = INIT_RECORDS.get(id(layer))
        if rec is None:
            raise ValueError(
                f"INIT_RECORDS has no entry for object id={id(layer)} ({type(layer).__name__}). "
                "Make sure you patched the class BEFORE building the model."
            )

        init_kwargs = self._bind_init_kwargs_from_record(layer, rec)

        # dtype/device align
        params_dtype = getattr(layer.config, "params_dtype", target_dtype)
        device = (
            layer.weight.device
            if getattr(layer, "weight", None) is not None
            else (torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu"))
        )
        W_small = W_small.to(device=device, dtype=params_dtype)

        # qkv sanity
        if projection_type == "qkv":
            out_dim = target_shape[0]  # ColumnParallelLinear: out_per_partition
            if (out_dim % 3) != 0 or (W_small.shape[0] % 3) != 0:
                raise ValueError(
                    f"QKV projection requires output dim and W_small rows divisible by 3, "
                    f"got out={out_dim}, W_small_rows={W_small.shape[0]}"
                )

        # inject projector args
        init_kwargs.update(
            {
                "W_small": W_small,
                "rank": rank,
                "projection_type": projection_type,
            }
        )

        # choose ctor and filter kwargs to its signature
        if isinstance(layer, ColumnParallelLinear):
            ctor = ColumnParallelLinearWithProjector
        elif isinstance(layer, RowParallelLinear):
            ctor = RowParallelLinearWithProjector
        else:
            raise TypeError(f"Unsupported layer type: {type(layer)}")

        ctor_kwargs = self._filter_kwargs_for_ctor(ctor, init_kwargs)

        # Ensure required keys are present (your error was exactly these missing)
        for req in ("input_size", "output_size"):
            if req not in ctor_kwargs or ctor_kwargs[req] is None:
                # take from live object as last resort
                ctor_kwargs[req] = getattr(layer, req)

        # build new layer
        new_layer = ctor(**ctor_kwargs)

        # copy bias if possible
        try:
            if getattr(layer, "bias", None) is not None and getattr(new_layer, "bias", None) is not None:
                if layer.bias.shape == new_layer.bias.shape:
                    with torch.no_grad():
                        new_layer.bias.data.copy_(layer.bias.data.to(dtype=params_dtype, device=device))
        except Exception as e:
            print(f"[warn] bias copy skipped: {e}")

        # swap
        setattr(parent_module, attr_name, new_layer)

        print(
            f"[Projector Replace] {name}: {type(layer).__name__} -> {type(new_layer).__name__} "
            f"(proj={projection_type}, rank={rank}, dtype={params_dtype}, device={device})"
        )

    def free_small_model(self):
        """
        Free the small model from memory and clear GPU cache if necessary.
        """
        if hasattr(self, "small_model"):
            print("Freeing small model and clearing GPU cache.")
            self.small_model = self.small_model.cpu()
            del self.small_model
            torch.cuda.empty_cache()

    def project_parameters(self, rank=64, learnable=False):
        small_state_dict = self.small_model.model.state_dict()
        large_num_layers = self.large_model.cfg.num_layers  # Assumes large_model has a cfg attribute

        # Step 1: Interpolate parameters between layers
        if hasattr(self.small_model.model.decoder, "layers") and hasattr(self.large_model.model.decoder, "layers"):
            small_layers = self.small_model.model.decoder.layers
            interpolated_params_list = self.expand_layers(small_layers, large_num_layers)

            # Step 2: Map interpolated parameters to large model layers and perform hidden_size projection
            for layer_idx, large_layer in enumerate(self.large_model.model.decoder.layers):

                print("Layer: " + str(layer_idx))
                interpolated_params = interpolated_params_list[layer_idx]

                for name, param_large in large_layer.named_parameters():
                    if name not in interpolated_params:
                        continue
                    print(name)

                    full_name = f"model.decoder.layers.{layer_idx}.{name}"
                    param_small = interpolated_params[name]

                    if param_small.shape == param_large.shape:
                        param_large.data.copy_(param_small)
                    elif len(param_small.shape) == 2:
                        if learnable:
                            module_path = name.split(".")[0]
                            self.replace_layer_with_projector(
                                parent_module=getattr(large_layer, module_path),
                                attr_name=name.split(".")[1],
                                W_small=param_small,
                                target_shape=param_large.shape,
                                target_dtype=self.dtype,
                                rank=rank,
                                name=name
                            )
                        else:
                            target_param = self.large_state_dict[full_name].to(param_small.device)
                            projected_param = self.dispatch_projection(
                                name=name,
                                W_small=param_small,
                                target_shape=param_large.shape,
                                target_param=target_param,
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

            if learnable:
                print("✅ Layer parameters projected successfully.")
                print("parameters in large model after layer projection:")
                for name, param in self.large_model.named_parameters():
                    print(f"{name}: {param.shape}")

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
                        target_param=param_large,
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
    utils.project_parameters(learnable=True)