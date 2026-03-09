from copy import copy
from enum import Enum, auto
from itertools import count

from kllm.sampling_parameters import SamplingParams

class SequenceStatus(Enum):
  WAITING   = auto()
  RUNNING   = auto()
  FINISHED  = auto()

class Sequence:
  '''
序列
存储一个推理任务的全部信息 (提示词 token, 生成的 token, 序列状态等)

block_size: 分页 KV Cache 中的页大小, 定义了每个内存块可以存储的 token 数量
counter:    用于保证序列 id 唯一的计数器

seq_id:     序列 id
status:     序列状态

token_ids:          推理任务的所有 token (提示词 token 和生成的 token)
last_token:         token_ids[-1]
num_tokens:         len(token_ids)
num_prompt_tokens:  提示词 token 的数量
num_cached_tokens:  已缓存 token 的数量
block_table:        分页 KV Cache 中的页表, 逻辑块到物理块的映射
'''
  block_size: int   = 256
  counter:    count = count()

  def __init__(
      self,
      token_ids:        list[int],
      sampling_params:  SamplingParams = SamplingParams(),
  ):
    self.seq_id = next(Sequence.counter)
    self.status = SequenceStatus.WAITING

    self.token_ids          = copy(token_ids) # 必须深拷贝, 保证不影响外部数据
    self.last_token         = token_ids[-1] if token_ids else None
    self.num_tokens         = len(token_ids)
    self.num_prompt_tokens  = len(token_ids)
    self.num_cached_tokens  = 0
    self.block_table        = []

    self.temperature  = sampling_params.temperature
    self.max_tokens   = sampling_params.max_tokens
    self.ignore_eos   = sampling_params.ignore_eos

  def __len__(self):
    return self.num_tokens

  def __getitem__(self, i: int):
    return self.token_ids[i]

  @property
  def is_finished(self):
    return self.status == SequenceStatus.FINISHED
  
  @property
  def num_completion_tokens(self):
    return self.num_tokens - self.num_prompt_tokens
  
  @property
  def prompt_token_ids(self):
    return self.token_ids[ : self.num_prompt_tokens]
  
  @property
  def completion_token_ids(self):
    return self.token_ids[self.num_prompt_tokens : ]
  
  @property
  def num_cached_blocks(self):
    return self.num_cached_tokens // self.block_size
  
  @property
  def num_blocks(self):
    return (self.num_tokens + self.block_size - 1) // self.block_size
  
  @property
  def last_block_num_tokens(self):
    return self.num_tokens - (self.num_blocks - 1) *  self.block_size

  def block(self, i: int):
    assert 0 <= i < self.num_blocks, "块下标越界"
    return self.token_ids[i * self.block_size : (i + 1) * self.block_size]
  
  def append_token(self, token_id: int):
    self.token_ids.append(token_id)
    self.last_token  = token_id
    self.num_tokens += 1

  def __getstate__(self):
    return (
      self.num_tokens,
      self.num_prompt_tokens,
      self.num_cached_tokens,
      self.block_table,
      self.token_ids if self.num_completion_tokens == 0 else self.last_token,
    )
  
  def __setstate__(self, state):
    (
      self.num_tokens,
      self.num_prompt_tokens,
      self.num_cached_tokens,
      self.block_table,
      last_token_or_ids
    ) = state
    if self.completion_token_ids == 0:
      self.token_ids = last_token_or_ids
    else:
      self.last_token = last_token_or_ids
