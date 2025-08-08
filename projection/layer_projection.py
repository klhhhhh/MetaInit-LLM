from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, linear_with_grad_accumulation_and_async_allreduce, linear_with_frozen_weight
from megatron.core.tensor_parallel.mapping import scatter_to_tensor_model_parallel_region, reduce_from_tensor_model_parallel_region

from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

class ColumnParallelLinearWithProjector(ColumnParallelLinear):
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

        self.projector_enabled = True
        self.projection_type = projection_type
        self.dtype = config.params_dtype

        self.register_buffer('W_small', W_small.clone().detach().to(self.dtype))

        d_out_large, d_in_large = self.output_size_per_partition, input_size
        d_out_small, d_in_small = W_small.shape

        if projection_type == "symmetric":
            self.A = nn.Parameter(torch.randn(d_out_large, rank, dtype=self.dtype) * 0.01)
            self.B = nn.Parameter(torch.randn(rank, d_out_small, dtype=self.dtype) * 0.01)

        elif projection_type == "asymmetric":
            self.A_out = nn.Parameter(torch.randn(d_out_large, rank, dtype=self.dtype) * 0.01)
            self.B_out = nn.Parameter(torch.randn(rank, d_out_small, dtype=self.dtype) * 0.01)
            self.A_in  = nn.Parameter(torch.randn(d_in_large, rank, dtype=self.dtype) * 0.01)
            self.B_in  = nn.Parameter(torch.randn(rank, d_in_small, dtype=self.dtype) * 0.01)

        elif projection_type == "qkv":
            # Assume each q/k/v small matrix is square
            d_total_small = W_small.shape[0]
            assert d_total_small % 3 == 0, "QKV projection requires W_small divisible by 3"
            d_single_small = d_total_small // 3
            d_single_large = d_out_large // 3

            self.q_proj = self._init_symmetric_proj(rank, d_single_large, d_single_small)
            self.k_proj = self._init_symmetric_proj(rank, d_single_large, d_single_small)
            self.v_proj = self._init_symmetric_proj(rank, d_single_large, d_single_small)

        else:
            raise ValueError(f"Unsupported projection_type: {projection_type}")

    def _init_symmetric_proj(self, rank, d_large, d_small):
        return nn.ParameterDict({
            'A': nn.Parameter(torch.randn(d_large, rank, dtype=self.dtype) * 0.01),
            'B': nn.Parameter(torch.randn(rank, d_small, dtype=self.dtype) * 0.01),
        })

    def get_projected_weight(self):
        W_s = self.W_small.to(self.dtype)

        if self.projection_type == "symmetric":
            P = self.A @ self.B
            return P @ W_s @ P.T

        elif self.projection_type == "asymmetric":
            P_out = self.A_out @ self.B_out
            P_in  = self.A_in @ self.B_in
            return P_out @ W_s @ P_in.T

        elif self.projection_type == "qkv":
            W_q, W_k, W_v = torch.chunk(W_s, 3, dim=0)

            def project(W, proj):
                P = proj['A'] @ proj['B']
                return P @ W @ P.T

            W_q_proj = project(W_q, self.q_proj)
            W_k_proj = project(W_k, self.k_proj)
            W_v_proj = project(W_v, self.v_proj)

            return torch.cat([W_q_proj, W_k_proj, W_v_proj], dim=0)

        else:
            raise NotImplementedError

    def forward(self, input_, weight=None, runtime_gather_output=None):

        if self.projector_enabled:
            weight = self.get_projected_weight()

        return super().forward(input_, weight=weight, runtime_gather_output=runtime_gather_output)

class RowParallelLinearWithProjector(RowParallelLinear):
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

        self.projector_enabled = True
        self.projection_type = projection_type
        self.dtype = config.params_dtype

        # Register small model weights as a buffer
        self.register_buffer("W_small", W_small.to(self.dtype))

        d_out_large, d_in_large = output_size, self.input_size_per_partition
        d_out_small, d_in_small = W_small.shape

        if projection_type == "symmetric":
            self.A = nn.Parameter(torch.randn(d_out_large, rank, dtype=self.dtype) * 0.01)
            self.B = nn.Parameter(torch.randn(rank, d_out_small, dtype=self.dtype) * 0.01)

        elif projection_type == "asymmetric":
            self.A_out = nn.Parameter(torch.randn(d_out_large, rank, dtype=self.dtype) * 0.01)
            self.B_out = nn.Parameter(torch.randn(rank, d_out_small, dtype=self.dtype) * 0.01)
            self.A_in  = nn.Parameter(torch.randn(d_in_large, rank, dtype=self.dtype) * 0.01)
            self.B_in  = nn.Parameter(torch.randn(rank, d_in_small, dtype=self.dtype) * 0.01)
        
        else:
            raise ValueError(f"Unsupported projection_type: {projection_type}")

    def get_projected_weight(self):
        W_s = self.W_small.to(self.dtype)

        if self.projection_type == "symmetric":
            P = self.A @ self.B
            return P @ W_s @ P.T

        elif self.projection_type == "asymmetric":
            P_out = self.A_out @ self.B_out
            P_in  = self.A_in @ self.B_in
            return P_out @ W_s @ P_in.T

        else:
            raise NotImplementedError

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

        # Construct projected weights
        with torch.autocast(device_type='cuda', dtype=self.config.params_dtype):
            W_proj = self.get_projected_weight()

        # Compute output
        if not self.weight.requires_grad:
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

        allreduce_dgrad = False

        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=W_proj,
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
