from typing import Callable, Type, Dict, Any
from torch import nn
from transformers import PretrainedConfig

_MODEL_REGISTRY: Dict[str, Callable[[PretrainedConfig], nn.Module]] = {}

def register_model(name: str):
  def decorator(model_class: Type[nn.Module]):
    _MODEL_REGISTRY[name] = model_class
    return model_class
  return decorator

def get_model(config: PretrainedConfig) -> nn.Module:
  model_type = getattr(config, "model_type", None)
  if model_type is None:
    raise ValueError(f"Config 中没有 model_type 属性")
  if model_type not in _MODEL_REGISTRY:
    raise ValueError(f"未知的模型类型: {model_type}. 支持的模型类型: {list(_MODEL_REGISTRY.keys())}")
  return _MODEL_REGISTRY[model_type](config)

def get_supported_models() -> list[str]:
  return list(_MODEL_REGISTRY.keys())

from vllm.models.qwen3 import Qwen3ForCausalLM
from vllm.models.gemma3 import Gemma3ForCausalLM
