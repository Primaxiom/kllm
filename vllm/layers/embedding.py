from math import min, max

import torch
import torch.nn.functional as F
from torch import nn, distributed as dist, Tensor as tensor

class VocabParallelEmbedding(nn.Module):
  def __init__(
    self,
    num_embeddings: int,
    embedding_size: int,
  ):
    super().__init__()
    self.tp_rank                = dist.get_rank()
    self.tp_size                = dist.get_world_size()
    self.embedding_size         = embedding_size
    self.num_embeddings         = num_embeddings
    self.num_padded_embeddings  = (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size
    self.num_shard_embeddings   = self.num_padded_embeddings // self.tp_size
    self.weight                 = nn.Parameter(torch.empty(self.num_shard_embeddings, embedding_size))
    self.weight.weight_loader   = self.weight_loader

  def weight_loader(
    self,
    param: nn.Parameter,
    loaded_weight: tensor,
  ):
    param_data    = param.data
    shard_offset  = self.num_shard_embeddings * self.tp_rank
    shard_size    = self.num_shard_embeddings

    start_index   = min(             shard_offset, self.num_embeddings)
    end_index     = min(shard_size + shard_offset, self.num_embeddings)
    shard_offset  = start_index
    shard_size    = end_index - start_index

    if shard_size > 0:
      sharded_weight = loaded_weight.narrow(0, shard_offset, shard_size)
      param_data[:shard_size].copy_(sharded_weight)

    if shard_size < self.num_shard_embeddings:
      param_data[:shard_size].zero_()

  def forward(self, x: tensor) -> tensor:
    shard_offset      = self.num_shard_embeddings * self.tp_rank
    shard_size        = self.num_shard_embeddings
    start_index:  int = min(             shard_offset, self.num_embeddings)
    end_index:    int = min(shard_size + shard_offset, self.num_embeddings)
    
    mask = (start_index <= x) & (x < end_index)
    x = mask * (x - start_index)
    output = F.embedding(x, self.weight)
    
    if dist.get_world_size() > 1:
      output = mask.unsqueeze(1) * output
      dist.all_reduce(output, op=dist.ReduceOp.SUM)

    return output

class ParallelLMHead(VocabParallelEmbedding):
  pass