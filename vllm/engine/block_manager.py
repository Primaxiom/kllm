from collections import deque

import xxhash
import numpy as np

from vllm.engine.sequence import Sequence

class Block:
  def __init__(
    self,
    block_id: int,
  ):
    self.block_id   = block_id
    self.hash       = -1
    self.ref_count  = 0
    self.token_ids  = []

  def update(
    self,
    hash:       int,
    token_ids:  list[int],
  ):
    self.hash       = hash
    self.token_ids  = token_ids

  def reset(self):
    self.hash       = -1
    self.ref_count  = 1
    self.token_ids  = []

class BlockManager:
  def __init__(
    self, 
    num_blocks: int, 
    block_size: int
  ):
    self.block_size:        int             = block_size
    self.blocks:            list[Block]     = [Block(i) for i in range(num_blocks)]
    self.hash_to_block_id:  dict[int, int]  = {}
    self.free_block_ids:    deque[int]      = deque(range(num_blocks))
    self.used_block_ids:    set[int]        = set()

  @classmethod
  def compute_hash(
    self, 
    token_ids:    list[int], 
    prefix_hash:  int = -1 ,
  ) -> int:
    h = xxhash.xxh64()
    if prefix_hash != -1:
      h.update(prefix_hash.to_bytes(8, "little"))
    h.update(np.array(token_ids, dtype=np.int32).tobytes())
    return h.intdigest()

  def _allocate_block(self, block_id: int) -> Block:
    block = self.blocks[block_id]
    assert block.ref_count == 0, f"内存块 {block_id} 已被分配"
    block.reset()
    self.free_block_ids.remove(block_id)
    self.used_block_ids.add   (block_id)
    return block
  
  def _deallocate_block(self, block_id: int):
    block = self.blocks[block_id]
    assert block.ref_count == 0, f"内存块 {block_id} 仍被引用"
    self.free_block_ids.append(block_id)
    self.used_block_ids.remove(block_id)

  def can_allocate(self, seq: Sequence) -> bool: 
    return seq.num_blocks <= len(self.free_block_ids)
  
  def allocate(self, seq: Sequence):
    assert seq.block_table == [], f"序列 {seq.seq_id} 页表非空"
    hash        = -1
    cache_miss  = False
    
    for logical_block_id in range(seq.num_blocks):

      token_ids         = seq.block(logical_block_id)
      hash              = self.compute_hash(token_ids, hash) if len(token_ids) == self.block_size else -1
      physical_block_id = self.hash_to_block_id.get(hash, -1)
      if physical_block_id == -1 or token_ids != self.blocks[physical_block_id].token_ids:
        # 前缀失配, 往后继续失配
        cache_miss = True

      if cache_miss:
        physical_block_id = self.free_block_ids[0]
        block             = self._allocate_block(physical_block_id)
      else:
        seq.num_cached_tokens += self.block_size
        if physical_block_id in self.used_block_ids:
          block                = self.blocks[physical_block_id]
          block.ref_count     += 1
        else:
          '''
虽然 physical_block_id not in self.used_block_ids 说明该块被 _deallocate_block, 
但是 not cache_miss 说明 physical_block_id != -1 且 token_ids == self.blocks[physical_block_id],
也即 self.hash_to_block_id 中存在的映射仍然有效, 可以复用
'''
          block = self._allocate_block(physical_block_id)

      if hash != -1:
        block.update(hash, token_ids)
        self.hash_to_block_id[hash] = physical_block_id

      seq.block_table.append(physical_block_id)

  def deallocate(self, seq: Sequence):
    for physical_block_id in seq.block_table:
      block = self.blocks[physical_block_id]
      assert block.ref_count > 0, f"内存块 {physical_block_id} 引用计数非正"
      block.ref_count -= 1
      if block.ref_count == 0:
        self._deallocate_block(physical_block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()

  def can_append(self, seq: Sequence) -> bool:
    # 若余 1, 则需要新块, 否则当前序列的最后一个 token 可以加到最后一个块
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

  def may_append(self, seq: Sequence):
    block_table             = seq.block_table
    last_physical_block_id  = seq.block_table[-1]
    last_block              = block_table[last_physical_block_id]

    if len(seq) % self.block_size == 1:
      assert last_block.hash != -1, f"内存块 {last_physical_block_id} 在完整时仍未计算哈希值"
      physical_block_id = self.free_block_ids[0]
      self._allocate_block(physical_block_id)
      seq.block_table.append(physical_block_id)

    elif len(seq) % self.block_size == 0:
      assert last_block.hash == -1, f"内存块 {last_physical_block_id} 在不完整时计算了哈希值"
      token_ids   = seq.block(seq.num_blocks - 1)
      prefix_hash = block_table[seq.block_table[-2]].hash if len(seq.block_table) > 1 else -1
      hash        = self.compute_hash(token_ids, prefix_hash)
      last_block.update(hash, token_ids)
      self.hash_to_block_id[hash] = last_physical_block_id

    else:
      assert last_block.hash == -1, f"内存块 {last_physical_block_id} 在不完整时计算了哈希值"
