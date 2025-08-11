from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, linear_with_grad_accumulation_and_async_allreduce, linear_with_frozen_weight
from megatron.core.tensor_parallel.mappings import scatter_to_tensor_model_parallel_region, reduce_from_tensor_model_parallel_region

from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

def _init_AB_pair(out_dim, in_dim_small, rank, device, dtype):
    A = nn.Parameter(torch.empty(out_dim, rank, device=device, dtype=dtype))
    B = nn.Parameter(torch.empty(rank, in_dim_small, device=device, dtype=dtype))
    # 简化的方差守恒：std ≈ 1/sqrt(fan)
    torch.nn.init.normal_(A, mean=0.0, std=1.0 / (in_dim_small ** 0.5))
    torch.nn.init.normal_(B, mean=0.0, std=1.0 / (rank ** 0.5))
    return A, B


# ===============================================
#  ColumnParallelLinear + Learnable Projector (α)
# ===============================================
class ColumnParallelLinearWithProjector(ColumnParallelLinear):
    """
    Extends ColumnParallelLinear with:
      - Learnable low-rank projection (symmetric / asymmetric / qkv)
      - Lazy initialization with one-time norm matching (proj_scale)
      - Learnable alpha (residual: W_eff = W_base + alpha * W_proj_scaled)
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config,
        init_method,
        W_small: torch.Tensor,
        rank: int = 64,
        projection_type: str = "symmetric",  # ["symmetric", "asymmetric", "qkv"]
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        embedding_activation_buffer=None,
        grad_output_buffer=None,
        is_expert=False,
        tp_comm_buffer_name=None,
        disable_grad_reduce=False,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=gather_output,
            stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            embedding_activation_buffer=embedding_activation_buffer,
            grad_output_buffer=grad_output_buffer,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            disable_grad_reduce=disable_grad_reduce,
        )

        assert projection_type in ("symmetric", "asymmetric", "qkv")
        self.projection_type = projection_type

        # Match device and dtype with the original layer
        device = self.weight.device
        dtype = self.config.params_dtype

        # Store the small model weight
        self.register_buffer("W_small", W_small.clone().detach().to(device=device, dtype=dtype), persistent=False)

        # Number of rows in the large weight for this partition (row-parallel: rows split across TP)
        d_out_large = self.output_size_per_partition
        d_in_large = self.input_size
        d_out_small, d_in_small = self.W_small.shape

        # Low-rank projection parameters
        if projection_type == "symmetric":
            # Only valid for square matrices (e.g., hidden -> hidden)
            self.A_out, self.B_out = _init_AB_pair(d_out_large, d_out_small, rank, device, dtype)
            # Symmetric in-side: reuse P for both in and out
            self.A_in = None
            self.B_in = None

        elif projection_type == "asymmetric":
            # Separate low-rank projections for out and in sides
            self.A_out, self.B_out = _init_AB_pair(d_out_large, d_out_small, rank, device, dtype)
            self.A_in,  self.B_in  = _init_AB_pair(d_in_large,  d_in_small,  rank, device, dtype)

        else:  # "qkv"
            # Separate projections for Q/K/V on the out side (symmetric), reuse P_in for in side
            # Simplified: qkv uses symmetric method (each segment is square)
            self.A_out, self.B_out = _init_AB_pair(d_out_large // 3, d_out_small // 3, rank, device, dtype)
            self.A_in = None
            self.B_in = None

        # Learnable alpha (residual weight), initialized small for stable training
        self.alpha = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))

        # Lazy initialization state & proj_scale (one-time norm matching)
        self.register_buffer("_proj_inited", torch.tensor(0, dtype=torch.int8), persistent=False)
        self.register_buffer("proj_scale", torch.tensor(1.0, device=device, dtype=dtype), persistent=False)

    # --- Internal: Compute projected weight based on type ---
    def _project_weight_once(self, input_dtype: torch.dtype) -> torch.Tensor:
        W_s = self.W_small.to(dtype=input_dtype)

        if self.projection_type == "symmetric":
            # P_out = A_out @ B_out  --> [d_out_large, d_out_small]
            P = (self.A_out @ self.B_out).to(dtype=input_dtype)
            # Symmetric: W_proj = P @ W_s @ P^T  --> [d_out_large, d_out_large]
            W_proj = P @ W_s @ P.transpose(-1, -2)

            # Ensure square mapping per partition: d_in_large == d_out_large
            if W_proj.shape != (self.output_size_per_partition, self.output_size_per_partition):
                raise RuntimeError(
                    f"[symmetric] requires square mapping per partition. "
                    f"got W_proj={tuple(W_proj.shape)}, expected=({self.output_size_per_partition}, "
                    f"{self.output_size_per_partition})"
                )
            # Ensure input_size matches output_size_per_partition (hidden)
            if self.input_size != self.output_size_per_partition:
                raise RuntimeError(
                    f"[symmetric] expects input_size == output_size_per_partition; "
                    f"got {self.input_size} vs {self.output_size_per_partition}"
                )

            # Target shape should be [out_per_partition, input_size]
            return W_proj  # Dimensions match, satisfies [out, in]

        elif self.projection_type == "asymmetric":
            # P_out: [d_out_large, d_out_small]
            P_out = (self.A_out @ self.B_out).to(dtype=input_dtype)
            # P_in : [d_in_large, d_in_small]
            P_in  = (self.A_in  @ self.B_in ).to(dtype=input_dtype)
            # W_proj: [d_out_large, d_in_large]
            return P_out @ W_s @ P_in.transpose(-1, -2)

        else:  # "qkv"
            # Split rows of the small model weight
            if (W_s.shape[0] % 3 != 0) or (self.output_size_per_partition % 3 != 0):
                raise RuntimeError(
                    f"[qkv] requires rows divisible by 3: W_small_rows={W_s.shape[0]}, "
                    f"out_per_partition={self.output_size_per_partition}"
                )
            d_out_s_per = W_s.shape[0] // 3
            d_out_l_per = self.output_size_per_partition // 3
            chunks_s = torch.chunk(W_s, 3, dim=0)

            # Symmetric: each segment uses the same (A_out, B_out) to generate P, then P @ W_s_i @ P^T
            P = (self.A_out @ self.B_out).to(dtype=input_dtype)  # [d_out_l_per, d_out_s_per]
            outs = []
            for i in range(3):
                W_i = chunks_s[i]
                W_i_proj = P @ W_i @ P.transpose(-1, -2)  # [d_out_l_per, d_out_l_per]
                # Same as symmetric, ensure square and in==out==hidden_per_partition
                if W_i_proj.shape != (d_out_l_per, d_out_l_per) or d_out_l_per != (self.input_size // 3):
                    # Raise error for slight mismatches to avoid silent shape errors
                    raise RuntimeError(
                        f"[qkv] per-chunk should be square and match in-dim per chunk. "
                        f"W_i_proj={tuple(W_i_proj.shape)}, out_per={d_out_l_per}, in_per={self.input_size//3}"
                    )
                outs.append(W_i_proj)

            return torch.cat(outs, dim=0)  # [out_per_partition, input_size]

    # --- Lazy initialization: one-time norm matching ---
    def _lazy_norm_init(self, input_dtype: torch.dtype):
        if int(self._proj_inited.item()) == 1:
            return
        with torch.no_grad():
            W_proj = self._project_weight_once(input_dtype)
            W_base = self.weight.to(dtype=input_dtype)

            eps = 1e-12
            norm_proj = torch.linalg.norm(W_proj, ord='fro')
            norm_base = torch.linalg.norm(W_base, ord='fro')
            scale = (norm_base / (norm_proj + eps)).detach()

            self.proj_scale.data = scale.to(dtype=self.proj_scale.dtype, device=self.proj_scale.device)
            # Keep alpha initialized to a small value (already set to 1e-3 in __init__)
            self._proj_inited.data.fill_(1)

    # --- Used in forward: Get the effective weight ---
    def _effective_weight(self, input_dtype: torch.dtype) -> torch.Tensor:
        # 1) Lazy initialization (one-time norm matching)
        self._lazy_norm_init(input_dtype)

        # 2) Generate W_proj and scale it
        W_proj = self._project_weight_once(input_dtype)
        W_proj = self.proj_scale.to(dtype=input_dtype) * W_proj

        # 3) Residual fusion: W_eff = W_base + alpha * W_proj
        W_base = self.weight.to(dtype=input_dtype)
        alpha  = self.alpha.to(dtype=input_dtype)
        return W_base + alpha * W_proj

    # --- Override forward, inject weight, reuse parent logic ---
    def forward(self, input_, weight=None, runtime_gather_output=None):
        # Ignore the passed weight, use our W_eff
        weight_override = self._effective_weight(input_.dtype)
        return super().forward(input_, weight=weight_override, runtime_gather_output=runtime_gather_output)


# =============================================
#  RowParallelLinear + Learnable Projector (α)
# =============================================
class RowParallelLinearWithProjector(RowParallelLinear):
    """
    Extends RowParallelLinear with:
      - Learnable low-rank projection (symmetric / asymmetric)
      - Lazy initialization with one-time norm matching (proj_scale)
      - Learnable alpha (residual: W_eff = W_base + alpha * W_proj_scaled)
    Notes:
      - The forward method of Row does not have a weight parameter. 
        Therefore, the original forward is strictly replicated here, 
        with only the weight used in matmul replaced by our W_eff.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        W_small: torch.Tensor,  # [output_size, input_size_small]
        rank: int = 64,
        projection_type: str = "symmetric",  # ["symmetric", "asymmetric", "qkv"]
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
        )

        assert projection_type in ("symmetric", "asymmetric")
        self.projection_type = projection_type

        device = self.weight.device
        dtype  = self.config.params_dtype

        self.register_buffer("W_small", W_small.clone().detach().to(device=device, dtype=dtype), persistent=False)

        d_out_large = self.output_size
        d_in_large  = self.input_size_per_partition  # Note: Row partitions along the column dimension
        d_out_small, d_in_small = self.W_small.shape

        if projection_type == "symmetric":
            # Only valid for square matrices (e.g., hidden -> hidden)
            self.A_out, self.B_out = _init_AB_pair(d_out_large, d_out_small, rank, device, dtype)
            self.A_in = None
            self.B_in = None
        else:
            self.A_out, self.B_out = _init_AB_pair(d_out_large, d_out_small, rank, device, dtype)
            # Row weight shape = [out, in_per_partition], we need to project the in side to in_per_partition
            self.A_in,  self.B_in  = _init_AB_pair(d_in_large,  d_in_small,  rank, device, dtype)

        self.alpha = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))
        self.register_buffer("_proj_inited", torch.tensor(0, dtype=torch.int8), persistent=False)
        self.register_buffer("proj_scale", torch.tensor(1.0, device=device, dtype=dtype), persistent=False)

    def _project_weight_once(self, input_dtype: torch.dtype) -> torch.Tensor:
        W_s = self.W_small.to(dtype=input_dtype)
        if self.projection_type == "symmetric":
            P = (self.A_out @ self.B_out).to(dtype=input_dtype)         # [d_out_large, d_out_small]
            W_proj_full = P @ W_s @ P.transpose(-1, -2)                 # [d_out_large, d_out_large]
            if W_proj_full.shape[1] != self.input_size_per_partition:
                raise RuntimeError(
                    f"[Row symmetric] expects in-per-partition == out; "
                    f"got in_per={self.input_size_per_partition}, out={self.output_size}"
                )
            # Slice/map to the current partition's columns (assuming no TP column splitting; adjust if needed)
            return W_proj_full[:, : self.input_size_per_partition]

        else:  # asymmetric
            P_out = (self.A_out @ self.B_out).to(dtype=input_dtype)     # [d_out_large, d_out_small]
            P_in  = (self.A_in  @ self.B_in ).to(dtype=input_dtype)     # [d_in_large,  d_in_small]
            W_proj = P_out @ W_s @ P_in.transpose(-1, -2)               # [d_out_large, d_in_large]
            return W_proj  # Its column dimension already equals in_per_partition

    def _lazy_norm_init(self, input_dtype: torch.dtype):
        if int(self._proj_inited.item()) == 1:
            return
        with torch.no_grad():
            W_proj = self._project_weight_once(input_dtype)
            W_base = self.weight.to(dtype=input_dtype)
            eps = 1e-12
            norm_proj = torch.linalg.norm(W_proj, ord='fro')
            norm_base = torch.linalg.norm(W_base, ord='fro')
            scale = (norm_base / (norm_proj + eps)).detach()
            self.proj_scale.data = scale.to(dtype=self.proj_scale.dtype, device=self.proj_scale.device)
            self._proj_inited.data.fill_(1)

    def _effective_weight(self, input_dtype: torch.dtype) -> torch.Tensor:
        self._lazy_norm_init(input_dtype)
        W_proj = self._project_weight_once(input_dtype)
        W_proj = self.proj_scale.to(dtype=input_dtype) * W_proj
        W_base = self.weight.to(dtype=input_dtype)
        alpha  = self.alpha.to(dtype=input_dtype)
        return W_base + alpha * W_proj

    # Strictly replicate the original Row forward, replacing the final weight used in matmul with our W_eff
    def forward(self, input_):
        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context is True:
                assert (
                    self.config.cpu_offloading is False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)

        # Determine _forward_impl
        if not self.weight.requires_grad:
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

        allreduce_dgrad = False

        # === Inject our weight ===
        weight_override = self._effective_weight(input_parallel.dtype)

        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=weight_override,   # <-- Use our constructed W_eff
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=allreduce_dgrad,
            sequence_parallel=False,
            grad_output_buffer=None,
            allreduce_dgrad=allreduce_dgrad,
        )

        if self.explicit_expert_comm:
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)

        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias

        return output, output_bias