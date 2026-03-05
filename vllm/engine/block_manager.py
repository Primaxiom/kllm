from collections import deque

import xxhash
import numpy as np

from vllm.engine.sequence import Sequence

class Block:
  def __init__(
    self,
    block_id: int,
  ):
    '''
ref_count > 0 时,
不应当存在 len(token_ids) != BlockManager.block_size 的块,
更不应当存在 len(token_ids) != BlockManager.block_size 且 hash != -1 的块,
也不应当存在 len(token_ids) == BlockManager.block_size 且 hash == -1 的块
'''
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
    '''
BlockManager 中涉及的 block_id 均为物理块号
hash_to_block_id 即为前缀哈希到物理块号的映射
'''
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
    '''
compute_hash

记字符串 s 的哈希值为 prefix_hash, 
  字符串 t 的 token 为 token_ids,
该函数计算字符串 s + t 的哈希值

这保证每个 Block 的哈希值包含前缀的所有内容 
'''
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
    '''
allocate

该函数为一整个 Sequence 的 token 分配空间, 
分配前应当保证其页表为空, 
即上一次分配的空间应当全部已经释放

序列是否有足够空间分配应当由调用者使用 BlockManager.can_allocate 进行检查

i 为逻辑块号, 在 Sequence 中的表现为连续存储, 使用 seq.block_table 转换成物理块号

对于每个完整块 i, 计算块 [0, i] 的 token 的哈希值作为该块的哈希值,
保证复用块时的 token 前缀一致,
不完整块的哈希值应当为 -1

若块 i 的哈希值若在 self.hash_to_block_id 中有记录则可无需再次分配, 
则只需增加引用计数,
否则分配 self.free_block_ids 队头元素为编号的内存块

若块 i 的哈希值有效, 
则更新 self.hash_to_block_id,
最后在 seq.block_table 记录物理块号
'''
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
此处是唯一会导致 deque 随机访问的情况, 不过一般情况下发生概率貌似很小 
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
    # 若 len(seq) % self.block_size 余 1, 则需要新块, 否则当前序列的最后一个 token 可以加到最后一个块
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

  def may_append(self, seq: Sequence):
    '''
may_append

decode 阶段使用的空间分配函数, 有新 token 时调用,
简化版的 allocate 函数, 讨论 3 种情况:
1. 该 token 不属于 last_block, 需要分配新块
2. 该 token 是 last_block 的最后一个 token, 更新 last_block 和 hash_to_block_id
3. 该 token 不是任何一个前述情况, 无需任何操作

每个块的分配和内容更新是分离的, 只有该块完整时才更新
'''
    block_table             = seq.block_table
    last_physical_block_id  = block_table[-1]
    last_block              = self.blocks[last_physical_block_id]

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
