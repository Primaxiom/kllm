import os
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass
class Config:
  model: str
  max_num_batched_tokens: int = 16384
  max_num_seqs: int = 512
  max_model_len: int = 4096
  gpu_memory_utilization: float = 0.9
  tensor_parallel_size: int = 1
  enforce_eager: bool = False
  hf_config: AutoConfig | None = None
  eos: int = -1
  kvcache_block_size: int = 256
  num_kvcache_blocks: int = -1

  def __post_init__(self):
    # assert os.path.isdir(self.model), f"模型路径 (\"{self.model}\") 不是一个有效路径"
    assert self.kvcache_block_size % 256 == 0, f"kvcache_block_size ({self.kvcache_block_size}) 无法被 256 整除"
    assert 1 <= self.tensor_parallel_size <= 8, f"tensor_parallel_size ({self.tensor_parallel_size}) 的值域应当为 [1, 8]"
    self.hf_config = AutoConfig.from_pretrained(self.model)
    self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
    assert self.max_num_batched_tokens >= self.max_model_len, f"批处理容量 ({self.max_num_batched_tokens}) 应当至少为模型上下文长度 ({self.max_model_len})"
