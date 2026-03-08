import os

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
  model_name = "gemma-3-1b-pt" # "Qwen/Qwen3-0.6B"
  path = os.path.expanduser(f"~/{model_name}/")
  hf_config = AutoConfig.from_pretrained(path)
  model = AutoModelForCausalLM.from_pretrained(path)
  tokenizer = AutoTokenizer.from_pretrained(path)
  print(hf_config)
  print(model)