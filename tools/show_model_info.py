from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
  model_name = "Qwen/Qwen3-0.6B"
  model = AutoModelForCausalLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  print(model)