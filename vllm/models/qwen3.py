import torch
from torch import nn, Tensor, distributed as dist
from transformers import Qwen3Config

from vllm.layers.activation import SiluAndMul
from vllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear, QKVColumnParallelLinear
from vllm.layers.layer_normalization import LayerNormalization
from vllm.layers.rotary_embedding import get_rope
from vllm.layers.attention import Attention
from vllm.layers.embedding import VocabParallelEmbedding, ParallelLMHead

class Qwen3MLP(nn.Module):
  def __init__(
    self,
    hidden_size:        int,
    intermediate_size:  int,
  ):
    super().__init__()
    self.gate_up    = MergedColumnParallelLinear(
      input_size    = hidden_size,
      output_sizes  = [intermediate_size] * 2,
    )
    self.act_fn     = SiluAndMul()
    self.down       = RowParallelLinear(
      input_size    = intermediate_size,
      output_size   = hidden_size
    )
  
  def forward(self, x: Tensor) -> Tensor:
    x = self.gate_up(x)
    x = self.act_fn(x)
    x = self.down(x)
    return x
  
class Qwen3Attention(nn.Module):
  def __init__(
    self,
    hidden_size:  int,
    num_heads:    int,
    num_kv_heads: int | None = None,
    scale:        float | None = None,
    qkv_bias:     bool = False,
    rms_norm_eps: float = 1e-06,
    rope_theta:   float = 10000,
    max_position: int = 128 * 1024,
  ):
    super().__init__()

    tp_size                 = dist.get_world_size()
    self.total_num_heads    = num_heads
    self.total_num_kv_heads = num_kv_heads if num_kv_heads else num_heads

    assert hidden_size              % num_heads               == 0
    assert num_heads                % self.total_num_kv_heads == 0
    assert self.total_num_kv_heads  % tp_size                 == 0

    self.head_dim     = hidden_size             // num_heads
    self.num_heads    = self.total_num_heads    // tp_size
    self.num_kv_heads = self.total_num_kv_heads // tp_size

    self.scale        = scale if scale else self.head_dim ** -0.5
    self.q_size       = self.head_dim * self.total_num_heads
    self.kv_size      = self.head_dim * self.total_num_kv_heads
    self.qkv_bias     = qkv_bias

    self.qkv          = QKVColumnParallelLinear(
      input_size      = hidden_size,
      head_size       = self.head_dim,
      num_heads       = self.total_num_heads,
      num_kv_heads    = self.total_num_kv_heads,
      bias            = self.qkv_bias
    )

    if not qkv_bias:
      self.q_norm     = LayerNormalization(self.head_dim, rms_norm_eps)
      self.k_norm     = LayerNormalization(self.head_dim, rms_norm_eps)

    self.rotary_emb   = get_rope(
      base            = rope_theta,
      embedding_dim   = self.head_dim,
      max_position    = max_position,
    )

    self.attention    = Attention(
      num_heads       = self.total_num_heads, 
      head_dim        = self.head_dim, 
      scale           = self.scale, 
      num_kv_heads    = self.total_num_kv_heads,
    )

    self.o            = RowParallelLinear(
      input_size      = hidden_size,
      output_size     = hidden_size
    )

  def forward(
    self,
    x:    Tensor,
    pos:  Tensor,
  ):
    qkv: Tensor = self.qkv(x)
    q, k, v     = qkv.split([self.q_size, self.kv_size, self.kv_size], -1)
    q           = q.view(-1, self.num_heads   , self.head_dim)
    k           = k.view(-1, self.num_kv_heads, self.head_dim)
    v           = v.view(-1, self.num_kv_heads, self.head_dim)

    if not self.qkv_bias:
      q, k      = self.q_norm(q), self.k_norm(k)

    q, k  = self.rotary_emb(pos, q, k)
    o     = self.attention(q, k, v)
    o     = self.o(o)

    return o

class Qwen3DecoderLayer(nn.Module):
  def __init__(
    self,
    cfg:  Qwen3Config
  ):
    super().__init__()
    self.rms_layernorm  = LayerNormalization(cfg.hidden_size, cfg.rms_norm_eps)
    self.gqa            = Qwen3Attention(
      cfg.hidden_size,
      cfg.num_attention_heads,
      cfg.num_key_value_heads,
      None,
      cfg.attention_bias,
      cfg.rms_norm_eps,
      cfg.rope_theta,
      cfg.max_position_embeddings,
    )
    self.mlp            = Qwen3MLP(
      cfg.hidden_size,
      cfg.intermediate_size,
    )
  
  def forward(
    self, 
    x:        Tensor,
    pos:      Tensor,
    residual: Tensor | None = None,
  ) -> tuple[Tensor, Tensor]:
    x, residual = self.rms_layernorm(x, residual) if residual else self.rms_layernorm(x), x
    x           = self.gqa(x, pos)
    x, residual = self.rms_layernorm(x, residual)
    x           = self.mlp(x)
    return x

class Qwen3Model(nn.Module):
  def __init__(
    self,
    cfg:  Qwen3Config
  ):
    super().__init__()
    self.embedding  = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
    self.layers     = nn.ModuleList([Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
    self.rms_norm   = LayerNormalization(cfg.hidden_size, cfg.rms_norm_eps)
    
  def forward(
    self,
    input_ids: Tensor,
    positions: Tensor,
  ) -> Tensor:
    x, residual = self.embedding(input_ids), None
    for layer in self.layers: x, residual = layer(x, positions, residual)
    x, residual = self.rms_norm(x, residual)
    return x
  
class Qwen3ForCausalLM(nn.Module):
  def __init__(
    self,
    cfg:  Qwen3Config
  ):
    super().__init__()
    self.model    = Qwen3Model(cfg)
    self.lm_head  = ParallelLMHead(cfg.vocab_size, cfg.hidden_size)
    if cfg.tie_word_embeddings:
      self.lm_head.weight.data = self.model.embedding.weight.data

  def forward(
    self,
    input_ids: Tensor,
    positions: Tensor,
  ) -> Tensor:
    return self.model(input_ids, positions)
  
  def compute_logits(self, x: Tensor) -> Tensor:
    return self.lm_head(x)
