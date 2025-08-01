from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ColumnParallelLinearWithProjector(ColumnParallelLinear):
    def __init__(
        self,
        input_size,
        output_size,
        *,
        config,
        init_method,
        W_small: torch.Tensor,  # Small model weights
        rank: int = 64,         # Projection rank
        symmetric: bool = True, # Projection method
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
        self.symmetric = symmetric
        self.dtype = config.params_dtype

        # Save the weights of the small model
        d_out_small, d_in_small = W_small.shape
        self.register_buffer('W_small', W_small.clone().detach().to(self.dtype))

        d_out_large, d_in_large = self.output_size_per_partition, input_size

        if symmetric:
            self.A = nn.Parameter(torch.randn(d_out_large, rank, dtype=self.dtype) * 0.01)
            self.B = nn.Parameter(torch.randn(rank, d_out_small, dtype=self.dtype) * 0.01)
        else:
            self.A_out = nn.Parameter(torch.randn(d_out_large, rank, dtype=self.dtype) * 0.01)
            self.B_out = nn.Parameter(torch.randn(rank, d_out_small, dtype=self.dtype) * 0.01)
            self.A_in = nn.Parameter(torch.randn(d_in_large, rank, dtype=self.dtype) * 0.01)
            self.B_in = nn.Parameter(torch.randn(rank, d_in_small, dtype=self.dtype) * 0.01)

    def get_projected_weight(self):
        W_s = self.W_small.to(self.dtype)

        if self.symmetric:
            P = self.A @ self.B  # (d_out_large, d_out_small)
            W_proj = P @ W_s @ P.T  # (d_out_large, d_in_large)
        else:
            P_out = self.A_out @ self.B_out  # (d_out_large, d_out_small)
            P_in = self.A_in @ self.B_in     # (d_in_large, d_in_small)
            W_proj = P_out @ W_s @ P_in.T    # (d_out_large, d_in_large)

        return W_proj

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

        dtype = config.params_dtype
        d_small_out, d_small_in = W_small.shape
        d_out_large, d_in_large = self.weight.shape

        # Initialize learnable projection matrices
        self.A_out = nn.Parameter(torch.randn(d_out_large, rank, dtype=dtype) * 0.01)
        self.B_out = nn.Parameter(torch.randn(rank, d_small_out, dtype=dtype) * 0.01)
        self.A_in = nn.Parameter(torch.randn(d_in_large, rank, dtype=dtype) * 0.01)
        self.B_in = nn.Parameter(torch.randn(rank, d_small_in, dtype=dtype) * 0.01)

        # Register small model weights as a buffer
        self.register_buffer("W_small", W_small.to(dtype))

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
            P_out = self.A_out @ self.B_out
            P_in = self.A_in @ self.B_in
            W_proj = P_out @ self.W_small @ P_in.transpose(-1, -2)

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
