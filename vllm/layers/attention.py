import torch
from torch import nn, Tensor
import triton
import triton.language as tl

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
  pass
