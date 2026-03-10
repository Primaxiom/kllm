from collections import deque
from typing import Optional

from kllm.config import Config
from kllm.engine.sequence import Sequence, SequenceStatus
from kllm.engine.block_manager import BlockManager

class Scheduler:
  def __init__(self, cfg: Config):
    self.eos                                = cfg.eos
    self.max_model_len                      = cfg.max_model_len
    self.max_num_seqs                       = cfg.max_num_seqs
    self.max_num_batched_tokens             = cfg.max_num_batched_tokens
    self.block_manager: BlockManager        = BlockManager(cfg.num_kvcache_blocks, cfg.kvcache_block_size)
    self.waiting_seqs:  deque[Sequence]     = deque[Sequence]()
    self.running_seqs:  deque[Sequence]     = deque[Sequence]()
    self.seqs:          dict[int, Sequence] = {}

  def is_finished(self) -> bool:
    return not self.waiting_seqs and not self.running_seqs
  
  def get_seq(self, seq_id: int) -> Optional[Sequence]:
    return self.seqs.get(seq_id, None)
  
  def add_seq(self, seq: Sequence):
    self.waiting_seqs.append(seq)
    self.seqs[seq.seq_id] = seq

  def finish_seq(self, seq: Sequence):
    if seq.status == SequenceStatus.FINISHED:
      return
    if seq.status == SequenceStatus.WAITING:
      self.waiting_seqs.remove(seq)
    elif seq.status == SequenceStatus.RUNNING:
      self.running_seqs.remove(seq)
    else:
      raise ValueError(f"未知的序列状态: {seq.status}")
    seq.status = SequenceStatus.FINISHED
    self.block_manager.deallocate(seq)
    self.seqs.pop(seq.seq_id, None)

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
    self.running_seqs.extendleft(reversed(scheduled_seqs))

    return scheduled_seqs, False

  def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
    for seq, token_id in zip(seqs, token_ids):
      seq.append_token(token_id)
      if not seq.ignore_eos and token_id == self.eos:
        finish_reason = "stop"
      elif seq.num_completion_tokens >= seq.max_tokens or seq.num_completion_tokens >= self.max_model_len:
        finish_reason = "length"
      else:
        finish_reason = None
      if finish_reason is not None:
        seq.finish_reason = finish_reason
        self.finish_seq(seq)
