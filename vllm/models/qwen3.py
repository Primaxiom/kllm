import torch
from torch import nn, Tensor

from vllm.layers.activation import SiluAndMul
from vllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear

class Qwen3MLP(nn.Module):
  def __init__(
    self,
    hidden_size:        int,
    intermediate_size:  int,
  ):
    super().__init__()
    self.gate_up    = MergedColumnParallelLinear(
      input_size    = hidden_size,
      output_sizes  = [intermediate_size] * 2,
    )
    self.act_fn     = SiluAndMul()
    self.down       = RowParallelLinear(
      input_size    = intermediate_size,
      output_size   = hidden_size
    )
  
  def forward(self, x: Tensor) -> Tensor:
    x = self.gate_up(x)
    x = self.act_fn(x)
    x = self.down(x)
    return x
  
