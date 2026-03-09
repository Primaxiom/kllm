import torch
from torch import nn, Tensor
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from kllm.utils.context import get_context

@triton.jit
def store_kvcache_kernel(
  key_ptr,
  key_stride,
  val_ptr,
  val_stride,
  k_cache_ptr,
  v_cache_ptr,
  slot_mapping_ptr,
  D: tl.constexpr,
):
  rank          = tl.program_id(0)
  slot          = tl.load(slot_mapping_ptr + rank)
  if slot == -1: return

  key_offsets   = key_stride * rank + tl.arange(0, D)
  val_offsets   = val_stride * rank + tl.arange(0, D)
  key           = tl.load(key_ptr + key_offsets)
  val           = tl.load(val_ptr + val_offsets)

  cache_offsets = slot * D + tl.arange(0, D)
  tl.store(k_cache_ptr + cache_offsets, key)
  tl.store(v_cache_ptr + cache_offsets, val)

def store_kvcache(
    key:          Tensor,
    val:          Tensor,
    k_cache:      Tensor,
    v_cache:      Tensor,
    slot_mapping: Tensor,
):
  num_tokens, num_heads, head_dim = key.shape
  D = num_heads * head_dim
  assert     key.stride(-1) == 1        and     val.stride(-1) == 1
  assert     key.stride( 1) == head_dim and     val.stride( 1) == head_dim
  assert k_cache.stride( 1) == D        and v_cache.stride( 1) == D
  assert slot_mapping.numel() == num_tokens
  grid = (num_tokens, 1)
  store_kvcache_kernel[grid](
    key, 
    key.stride(0),
    val,
    val.stride(0),
    k_cache,
    v_cache,
    slot_mapping,
    D,
  )

class Attention(nn.Module):
  def __init__(
    self,
    num_heads:    int,
    head_dim:     int,
    scale:        float           = 1.0,
    num_kv_heads: int             = None,
    window_size:  tuple[int, int] = (-1, -1),
  ):
    super().__init__()
    self.num_heads    = num_heads
    self.head_dim     = head_dim
    self.scale        = scale
    self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
    self.window_size  = window_size
    self.k_cache      = self.v_cache  \
                      = torch.tensor([])
  
  def forward(
      self,
      q: Tensor,
      k: Tensor,
      v: Tensor,
  ):
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache
    assert context and (context.slot_mapping is not None)
    if  k_cache.numel() > 0 and v_cache.numel() > 0:
      store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
    
    if context.is_prefill:
      if context.block_tables is not None:
        k, v = k_cache, v_cache
      o = flash_attn_varlen_func(
        q, k, v,
        max_seqlen_q=context.max_seqlen_q, 
        cu_seqlens_q=context.cu_seqlens_q,
        max_seqlen_k=context.max_seqlen_k, 
        cu_seqlens_k=context.cu_seqlens_k,
        softmax_scale=self.scale, 
        causal=True, 
        block_table=context.block_tables,
        window_size=self.window_size,
      )
    else:
      o = flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache, v_cache,
        cache_seqlens=context.context_lens, 
        block_table=context.block_tables, 
        softmax_scale=self.scale, 
        causal=True,
        window_size=self.window_size,
      )
    return o
