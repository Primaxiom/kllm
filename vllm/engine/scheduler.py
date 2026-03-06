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
    scheduled_seqs      = []
    num_batched_tokens  = 0

    while self.waiting_seqs \
      and self.max_num_seqs > len(scheduled_seqs):
      seq = self.waiting_seqs[0]
      if not self.block_manager.can_allocate(seq) \
          or self.max_num_batched_tokens < len(seq) + num_batched_tokens - seq.num_cached_tokens:
        break
      self.block_manager.allocate(seq)
      self.waiting_seqs.popleft()
      self.running_seqs.append(seq)
      seq.status = SequenceStatus.RUNNING
      scheduled_seqs.append(seq)
      num_batched_tokens += len(seq) - seq.num_cached_tokens
    if scheduled_seqs:
      return scheduled_seqs, True
    
    while self.running_seqs \
      and self.max_num_seqs > len(scheduled_seqs):
      seq = self.running_seqs.popleft()
      while not self.block_manager.can_append(seq):
        if self.running_seqs:
          self.preempt(self.running_seqs.pop())
        else:
          self.preempt(seq)
          break
      else:
        scheduled_seqs.append(seq)
        self.block_manager.may_append(seq)
    assert scheduled_seqs, "无任何序列可调度"
    self.running_seqs.appendleft(reversed(scheduled_seqs))
    
    return scheduled_seqs, False

  def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
    pass
