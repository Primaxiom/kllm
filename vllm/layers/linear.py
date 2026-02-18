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
    super().__init__()
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
  
  def weight_loader(self, param: nn.Parameter, loaded_weight: tensor):
    raise NotImplementedError("未实现 Linear::weight_loader()")
    
  def forward(self, x: tensor) -> tensor:
    raise NotImplementedError("未实现 Linear::forward()")

class RowParallelLinear(Linear):
  def __init__(
      self,
      input_size:   int,
      output_size:  int,
      bias:         bool = False,
  ):
    tp_size = dist.get_world_size()
    assert input_size % tp_size == 0, "input_size 必须能被 tp_size 整除"
    super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

  def weight_loader(self, param: nn.Parameter, loaded_weight: tensor):
    param_data    = param.data
    shard_size    = param_data.size(1)
    total_size    = loaded_weight.size(1)
    assert shard_size * self.tp_size == total_size, "加载的权重形状错误"
    start_index   = self.tp_rank * shard_size
    sliced_weight = loaded_weight.narrow(1, start_index, shard_size)
    param_data.copy_(sliced_weight)

  def forward(self, x: tensor) -> tensor:
    result = nn.functional.linear(x, self.weight, self.bias)
    if self.tp_size > 1:
      dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result

class ColumnParallelLinear(Linear):
  def __init__(
      self,
      input_size:   int,
      output_size:  int,
      bias:         bool = False,
  ):
    tp_size = dist.get_world_size()
    assert output_size % tp_size == 0, "output_size 必须能被 tp_size 整除"
    super().__init__(input_size, output_size // tp_size, bias, tp_dim=0)
  
  def weight_loader(self, param: nn.Parameter, loaded_weight: tensor):
    param_data    = param.data
    shard_size    = param_data.size(0)
    total_size    = loaded_weight.size(0)
    assert shard_size * self.tp_size == total_size, "加载的权重形状错误"
    start_index   = self.tp_rank * shard_size
    sliced_weight = loaded_weight.narrow(0, start_index, shard_size)
    param_data.copy_(sliced_weight)

  def forward(self, x: tensor) -> tensor:
    return nn.functional.linear(x, self.weight, self.bias)

class MergedColumnParallelLinear(ColumnParallelLinear):
  def __init__(
      self,
      input_size:   int,
      output_sizes: list[int],
      bias:         bool = False
  ):
    self.output_sizes = output_sizes
    super().__init__(input_size, sum(output_sizes), bias)

  def weight_loader(
    self, 
    param:            nn.Parameter, 
    loaded_weight:    tensor, 
    loaded_shard_id:  int
  ):
    shard_offset  = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
    shard_size    = self.output_sizes[loaded_shard_id] // self.tp_size
    param_data    = param.data.narrow(0, shard_offset, shard_size)
    start_index   = self.tp_rank * shard_size
    sliced_weight = loaded_weight.narrow(0, start_index, shard_size)
    param_data.copy_(sliced_weight)

class QKVColumnParallelLinear(ColumnParallelLinear):
  def __init__(
    self,
    input_size:   int, 
    head_size:    int,
    num_heads:    int,
    num_kv_heads: int | None  = None,
    bias:         bool        = False
  ):
    self.tp_size      = dist.get_world_size()
    if num_kv_heads is None: num_kv_heads = num_heads
    assert num_heads    % self.tp_size == 0, "num_heads 必须能被 tp_size 整除"
    assert num_kv_heads % self.tp_size == 0, "num_kv_heads 必须能被 tp_size 整除"
    self.num_heads    = num_heads     // self.tp_size
    self.num_kv_heads = num_kv_heads  // self.tp_size
    self.head_size    = head_size
    output_size       = head_size * (num_heads + 2 * num_kv_heads)
    self.output_size  = output_size // self.tp_size
    super().__init__(input_size, output_size, bias)

  def weight_loader(
    self, 
    param_data:       nn.Parameter, 
    loaded_weight:    tensor, 
    loaded_weight_id: str
  ):
    assert loaded_weight_id in ["q", "k", "v"]
    if    loaded_weight_id == "q":
      shard_offset  = 0
      shard_size    = self.head_size * self.num_heads
    elif  loaded_weight_id == "k":
      shard_offset  = self.head_size * self.num_heads
      shard_size    = self.head_size * self.num_kv_heads
    elif  loaded_weight_id == "v":
      shard_offset  = self.head_size * (self.num_heads + self.num_kv_heads)
      shard_size    = self.head_size * self.num_kv_heads
    else:
      raise ValueError(f"loaded_weight_id 有误: {loaded_weight_id}")
    
    param_data    = param_data.narrow(0, shard_offset, shard_size)
    start_index   = self.tp_rank * shard_size
    shared_weight = loaded_weight.narrow(0, start_index, shard_size)
    param_data.copy_(shared_weight)

if __name__ == "__main__":
  if dist.is_available() and not dist.is_initialized():
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://localhost:29500",
        rank=0,
        world_size=1,
    )
  layer = Linear(input_size=10, output_size=5)
  print("Linear layer initialized:", layer)
