import torch
from torch import nn, distributed as dist, Tensor as tensor

class Linear(nn.Module):
  def __init__(
      self,
      input_size:   int,
      output_size:  int,
      bias:         bool        = False,
      tp_dim:       int | None  = None,
  ):
    super.__init__()
    self.tp_dim         = tp_dim
    self.tp_rank        = dist.get_rank()
    self.tp_size        = dist.get_world_size()
    self.weight         = nn.Parameter(torch.empty(output_size, input_size))
    self.weight \
        .weight_loader  = self.weight_loader
    if bias:
      self.bias         = nn.Parameter(torch.empty(output_size))
      self.bias \
        .weight_loader  = self.weight_loader
    else:
      self.register_parameter("bias", None)
    
  def forward(self, x: tensor) -> tensor:
    raise NotImplementedError("未实现 Linear::forward()")
  
  def weight_loader(self, param: nn.Parameter, loaded_weight: tensor):
    raise NotImplementedError("未实现 Linear::weight_loader()")

class RowParallelLinear(Linear):
  pass

class ColumnParallelLinear(Linear):
  pass

class MergedColumnParallelLinear(ColumnParallelLinear):
  pass

class QKVColumnParallelLinear(ColumnParallelLinear):
  pass

if __name__ == "__main__":
  pass
