from collections import deque

from vllm.config import Config
from vllm.engine.sequence import Sequence, SequenceStatus
from vllm.engine.block_manager import BlockManager

class Scheduler:
  def __init__(self, cfg: Config):
    self.eos                            = cfg.eos
    self.max_num_seqs                   = cfg.max_num_seqs
    self.max_num_batched_tokens         = cfg.max_num_batched_tokens
    self.block_manager                  = BlockManager(cfg.num_kvcache_blocks, cfg.kvcache_block_size)
    self.waiting_seqs: deque[Sequence]  = deque()
    self.running_seqs: deque[Sequence]  = deque()

  def is_finished(self) -> bool:
    return not self.waiting_seqs and not self.running_seqs
  
  def add_seq(self, seq: Sequence):
    self.waiting_seqs.append(seq)

  def preempt(self, seq: Sequence):
    assert seq.status == SequenceStatus.RUNNING, f"序列 {seq.seq_id} 在非 RUNNING 状态下被抢占"
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)
    self.waiting_seqs.appendleft(seq)

  def schedule(self) -> tuple[list[Sequence], bool]:
    pass

  def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
    pass
