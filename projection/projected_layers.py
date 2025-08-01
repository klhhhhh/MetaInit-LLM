from megatron.core.tensor_parallel.layers import ColumnParallelLinear
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
