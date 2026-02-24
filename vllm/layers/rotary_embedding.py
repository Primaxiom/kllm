import torch
from torch import nn, Tensor

def apply_rotary_emb(
    x:    Tensor,
    cos:  Tensor,
    sin:  Tensor,
) -> Tensor:
  x1, x2  = torch.chunk(x, 2, -1)
  y1      = x1 * cos - x2 * sin
  y2      = x1 * sin + x2 * cos
  return torch.cat((y1, y2), -1)

class RotaryEmbedding(nn.Module):
  def __init__(
    self,
    base:           float,
    embedding_dim:  int, 
    max_position:   int = 2048
  ):
    super().__init__()
    inv_freq  = 1.0 / (base ** (torch.arange(0, embedding_dim, 2, dtype=torch.float) / embedding_dim))
    t         = torch.arange(max_position, dtype=torch.float)
    freqs     = torch.einsum("i,j -> ij", t, inv_freq)
    cos       = freqs.cos()
    sin       = freqs.sin()
    cache     = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
    self.register_buffer("cos_sin_cache", cache, persistent=False)
    self.embedding_dim = embedding_dim

  def forward(
    self,
    pos:  Tensor,
    qry:  Tensor,
    key:  Tensor,
  ) -> Tensor:
    cos, sin = self.cos_sin_cache[pos].chunk(2, -1)
    qry = apply_rotary_emb(qry, cos, sin)
    key = apply_rotary_emb(key, cos, sin)
    return qry, key
