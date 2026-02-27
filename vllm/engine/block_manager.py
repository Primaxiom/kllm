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
  pass
