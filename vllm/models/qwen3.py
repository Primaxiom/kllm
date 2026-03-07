if __name__ == "__main__":
  import sys, pathlib
  sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

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
    self.gate_up_proj = MergedColumnParallelLinear(
      input_size      = hidden_size,
      output_sizes    = [intermediate_size] * 2,
    )
    self.act_fn       = SiluAndMul()
    self.down_proj    = RowParallelLinear(
      input_size      = intermediate_size,
      output_size     = hidden_size
    )
  
  def forward(self, x: Tensor) -> Tensor:
    x = self.gate_up_proj(x)  
    x = self.act_fn(x)
    x = self.down_proj(x)
    return x
  
class Qwen3Attention(nn.Module):
  def __init__(
    self,
    hidden_size:  int,
    num_heads:    int,
    num_kv_heads: int | None = None,
    head_dim:     int | None = None,
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

    self.head_dim     = head_dim or hidden_size // num_heads
    self.num_heads    = self.total_num_heads    // tp_size
    self.num_kv_heads = self.total_num_kv_heads // tp_size

    self.scale        = scale if scale else self.head_dim ** -0.5
    self.q_size       = self.head_dim * self.num_heads
    self.kv_size      = self.head_dim * self.num_kv_heads
    self.qkv_bias     = qkv_bias

    self.qkv_proj     = QKVColumnParallelLinear(
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

    self.o_proj       = RowParallelLinear(
      input_size      = self.total_num_heads * self.head_dim,
      output_size     = hidden_size,
    )

  def forward(
    self,
    x:    Tensor,
    pos:  Tensor,
  ):
    qkv: Tensor = self.qkv_proj(x)
    q, k, v     = qkv.split([self.q_size, self.kv_size, self.kv_size], -1)
    q_proj      = q.view(-1, self.num_heads   , self.head_dim)
    k_proj      = k.view(-1, self.num_kv_heads, self.head_dim)
    v_proj      = v.view(-1, self.num_kv_heads, self.head_dim)

    if not self.qkv_bias:
      q_proj, k_proj = self.q_norm(q_proj), self.k_norm(k_proj)

    q_proj, k_proj   = self.rotary_emb(pos, q_proj, k_proj)
    o     = self.attention(q_proj, k_proj, v_proj)
    o     = self.o_proj(o.flatten(1, -1))

    return o

class Qwen3DecoderLayer(nn.Module):
  def __init__(
    self,
    cfg:  Qwen3Config
  ):
    super().__init__()
    self.input_layernorm  = LayerNormalization(cfg.hidden_size, cfg.rms_norm_eps)
    self.self_attn        = Qwen3Attention(
      cfg.hidden_size,
      cfg.num_attention_heads,
      cfg.num_key_value_heads,
      cfg.head_dim,
      None,
      cfg.attention_bias,
      cfg.rms_norm_eps,
      cfg.rope_theta if hasattr(cfg, "rope_theta") else cfg.rope_parameters["rope_theta"],
      cfg.max_position_embeddings,
    )
    self.mlp            = Qwen3MLP(
      cfg.hidden_size,
      cfg.intermediate_size,
    )
    self.post_attention_layernorm = LayerNormalization(cfg.hidden_size, cfg.rms_norm_eps)
  
  def forward(
    self, 
    x:        Tensor,
    pos:      Tensor,
    residual: Tensor | None = None,
  ) -> tuple[Tensor, Tensor]:
    x, residual = self.input_layernorm(x, residual) if residual is not None else (self.input_layernorm(x), x)
    x           = self.self_attn(x, pos)
    x, residual = self.post_attention_layernorm(x, residual)
    x           = self.mlp(x)
    return x, residual

class Qwen3Model(nn.Module):
  def __init__(
    self,
    cfg:  Qwen3Config
  ):
    super().__init__()
    self.embed_tokens   = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
    self.layers         = nn.ModuleList([Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
    self.norm           = LayerNormalization(cfg.hidden_size, cfg.rms_norm_eps)
    
  def forward(
    self,
    input_ids: Tensor,
    positions: Tensor,
  ) -> Tensor:
    x, residual = self.embed_tokens(input_ids), None
    for layer in self.layers: x, residual = layer(x, positions, residual)
    x, residual = self.norm(x, residual)
    return x
  
class Qwen3ForCausalLM(nn.Module):
  packed_modules_mapping: dict[str, tuple[str, int | str]] = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
  }

  def __init__(
    self,
    cfg:  Qwen3Config
  ):
    super().__init__()
    self.model    = Qwen3Model(cfg)
    self.lm_head  = ParallelLMHead(cfg.vocab_size, cfg.hidden_size)
    if cfg.tie_word_embeddings:
      self.lm_head.weight.data = self.model.embed_tokens.weight.data

  def forward(
    self,
    input_ids: Tensor,
    positions: Tensor,
  ) -> Tensor:
    return self.model(input_ids, positions)
  
  def compute_logits(self, x: Tensor) -> Tensor:
    return self.lm_head(x)

if __name__ == "__main__":
  if not dist.is_initialized():
    dist.init_process_group(
      backend="gloo", 
      init_method="tcp://127.0.0.1:29500?use_libuv=False",
      rank=0,
      world_size=1,
    )
  cfg = Qwen3Config(
    vocab_size=50257,
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=2,
    num_attention_heads=12,
    num_key_value_heads=12,
    head_dim=64,
  )
  model = Qwen3ForCausalLM(cfg)
  model = model.cuda()
  input_ids = torch.randint(0, 50257, (16,)).cuda()
  positions = torch.arange(16).cuda()
  output = model(input_ids, positions)
