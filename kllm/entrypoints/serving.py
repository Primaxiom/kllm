import time

from kllm.llm import LLM
from kllm.entrypoints.protocol import CompletionRequest
from kllm.sampling_parameters import SamplingParams

class OpenAIServing:
  def __init__(self, engine: LLM, model_name: str):
    self.engine     = engine
    self.model_name = model_name
    self.tokenizer  = engine.tokenizer
  
  def _extract_sampling_params(self, request: CompletionRequest) -> SamplingParams:
    sampling_params = SamplingParams()
    for key, value in request:
      if (value is not None) and hasattr(sampling_params, key):
        setattr(sampling_params, key, value)
    return sampling_params

  async def _generate_full(
    self, prompt: str, sampling_params: SamplingParams, request_id: str
  ):
    text_outputs              = ["" for _ in range(1)]
    finish_reason: str | None = None
    num_prompt_tokens         = 0
    num_completion_tokens     = 0
    print(f"开始生成")
    print(f"Prompt: {prompt}")
    print(f"Completion: ", end="", flush=True)
    async for res in self.engine.generate(prompt, sampling_params, request_id):
      for i in range(1):
        token_str        = res.token_str
        text_outputs[i] += token_str
      if res.is_finished:
        finish_reason         = res.finish_reason.lower() if res.finish_reason else None
        num_prompt_tokens     = res.num_prompt_tokens
        num_completion_tokens = res.num_completion_tokens
    return text_outputs, finish_reason, num_prompt_tokens, num_completion_tokens
  