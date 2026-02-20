from math import min, max

import torch
import torch.nn.functional as F
from torch import nn, distributed as dist, Tensor

class VocabParallelEmbedding(nn.Module):
  '''
并行词嵌入层
embedding_size / embedding_dim: 嵌入空间的维数
num_embeddings:                 词表长度
num_padded_embeddings:          填充后的词表长度, 使得该长度可以被 tp_size 整除
num_shard_embeddings:           切分后的词表长度
'''
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

  '''
加载权重:
由于词表被填充, 因此需要先计算实际的开始和结束位置
拷贝实际涉及的数据, 填充部分使用全零占位
'''
  def weight_loader(
    self,
    param:          nn.Parameter,
    loaded_weight:  Tensor,
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

  '''
输入: token 序列 (一个整数张量)
前向: 
  计算 mask, 只根据该设备上有的切分词表进行嵌入, [start_index, end_index) 之外的需要被 mask
  嵌入前应当先统一将 x 中的各元素减去 start_index, 因为切分词表的 0 对应 start_index
  每份输出的各元素应当再次和广播后的 mask 各元素相乘, 最后相加得到最终输出
'''
  def forward(self, x: Tensor) -> Tensor:
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
  '''
并行语言模型头
权重加载的模式相同, 故进行继承 (weight tying(?))
输入: prefill 时为一批完整 token 序列, decode 时为一批末尾 token 
前向: prefill 时会先根据 seqlens 提取末尾 token, 然后进行 linear, 最后将各部分 logits 拼接
输出: 下一 token 的 logits
'''
  def __init__(
    self, 
    num_embeddings: int, 
    embedding_size: int,
  ):
    super().__init__(num_embeddings, embedding_size)

  def forward(self, x: Tensor) -> Tensor:
    context = NotImplemented # TODO: get_context()
    if context.is_prefill:
      last_indices = context.cu_seqlens_q[1:] - 1
      x = x[last_indices].contiguous()
    logits = F.linear(x, self.weight)
    if self.tp_size > 1:
      all_logits = [
        torch.empty_like(logits) 
        for _ in range(self.tp_size)
      ] \
      if self.tp_rank == 0 \
      else None
      dist.gather(logits, all_logits, 0)
      logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
    return logits