if __name__ == "__main__":
  import sys, pathlib
  sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import torch
from torch import nn, Tensor, distributed as dist
from transformers import Gemma3TextConfig

from kllm.layers.activation import GELUTanhAndMul
from kllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear, QKVColumnParallelLinear
from kllm.layers.layer_normalization import GemmaRMSNorm
from kllm.layers.rotary_embedding import get_rope
from kllm.layers.attention import Attention
from kllm.layers.embedding import VocabParallelEmbedding, ParallelLMHead
from kllm.models import register_model

class Gemma3MLP(nn.Module):
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
    self.act_fn       = GELUTanhAndMul()
    self.down_proj    = RowParallelLinear(
      input_size      = intermediate_size,
      output_size     = hidden_size
    )
  
  def forward(self, x: Tensor) -> Tensor:
    x = self.gate_up_proj(x)  
    x = self.act_fn(x)
    x = self.down_proj(x)
    return x
  
class Gemma3Attention(nn.Module):
  def __init__(
    self,
    hidden_size:    int,
    num_heads:      int,
    num_kv_heads:   int | None = None,
    head_dim:       int | None = None,
    scale:          float | None = None,
    qkv_bias:       bool = False,
    rms_norm_eps:   float = 1e-06,
    rope_theta:     float = 10000,
    max_position:   int = 128 * 1024,
    sliding_window: int | None = None,
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

    self.scale        = scale if scale is not None else self.head_dim ** -0.5
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
      self.q_norm     = GemmaRMSNorm(self.head_dim, rms_norm_eps)
      self.k_norm     = GemmaRMSNorm(self.head_dim, rms_norm_eps)

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
      window_size     = (sliding_window, 0) if (sliding_window is not None) else (-1, -1)
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

class Gemma3DecoderLayer(nn.Module):
  def __init__(
    self,
    cfg:        Gemma3TextConfig,
    layer_type: str,
  ):
    super().__init__()
    self.input_layernorm  = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    self.self_attn        = Gemma3Attention(
      cfg.hidden_size,
      cfg.num_attention_heads,
      cfg.num_key_value_heads,
      cfg.head_dim,
      cfg.query_pre_attn_scalar ** -0.5,
      cfg.attention_bias,
      cfg.rms_norm_eps,
      cfg.rope_parameters[layer_type]["rope_theta"],
      cfg.max_position_embeddings,
      cfg.sliding_window if layer_type == "sliding_attention" else None,
    )
    self.post_attention_layernorm   = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    self.pre_feedforward_layernorm  = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    self.mlp            = Gemma3MLP(
      cfg.hidden_size,
      cfg.intermediate_size,
    )
    self.post_feedforward_layernorm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
  
  def forward(
    self, 
    x:        Tensor,
    pos:      Tensor,
    residual: Tensor | None = None,
  ) -> tuple[Tensor, Tensor]:
    x, residual = self.input_layernorm(x, residual) if residual is not None else (self.input_layernorm(x), x)
    x           = self.self_attn(x, pos)
    x           = self.post_attention_layernorm(x)
    x, residual = self.pre_feedforward_layernorm(x, residual)
    x           = self.mlp(x)
    x           = self.post_feedforward_layernorm(x)
    return x, residual

class Gemma3Model(nn.Module):
  def __init__(
    self,
    cfg:  Gemma3TextConfig
  ):
    super().__init__()
    self.embed_tokens   = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
    self.layers         = nn.ModuleList([Gemma3DecoderLayer(cfg, layer_type) for layer_type in cfg.layer_types])
    self.norm           = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    normalizer          = cfg.hidden_size ** 0.5
    self.register_buffer("normalizer", torch.tensor(normalizer), persistent=False)

  def forward(
    self,
    input_ids: Tensor,
    positions: Tensor,
  ) -> Tensor:
    x = self.embed_tokens(input_ids)
    x, residual = x * self.normalizer, None
    for layer in self.layers: x, residual = layer(x, positions, residual)
    x, residual = self.norm(x, residual)
    return x

@register_model("gemma3_text")
class Gemma3ForCausalLM(nn.Module):
  packed_modules_mapping: dict[str, tuple[str, int | str]] = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
  }
  chat_template: str = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() + eos_token }}{% endif %}{% endfor %}"

  def __init__(
    self,
    cfg:  Gemma3TextConfig
  ):
    super().__init__()
    self.model    = Gemma3Model(cfg)
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
