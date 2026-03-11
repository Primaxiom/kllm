import time
from typing import AsyncGenerator

from fastapi.responses import StreamingResponse

from kllm.llm import LLM
from kllm.entrypoints.protocol import CompletionRequest, CompletionResponse, CompletionResponseChoice, UsageInfo
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
    async for res in self.engine.generate(prompt, sampling_params, request_id):
      for i in range(1):
        token_str        = res.token_str
        text_outputs[i] += token_str
      if res.is_finished:
        finish_reason         = res.finish_reason.lower() if res.finish_reason else None
        num_prompt_tokens     = res.num_prompt_tokens
        num_completion_tokens = res.num_completion_tokens
    return text_outputs, finish_reason, num_prompt_tokens, num_completion_tokens
  
class OpenAIServingCompletion(OpenAIServing):
  async def create_completion(self, request: CompletionRequest):
    create_time_ns  = time.time_ns()
    create_time_sec = create_time_ns // 1_000_000_000
    request_id      = f"cmpl-{create_time_ns}"
    sampling_params = self._extract_sampling_params(request)
    prompt          = request.prompt

    if request.stream:
      return StreamingResponse(
        self.completion_stream_generator(
          request,
          prompt,
          request_id,
          create_time_sec,
        ),
        media_type="text/event-stream",
      )
    
    text_outputs, finish_reason, num_prompt_tokens, num_completion_tokens = await self._generate_full(prompt, sampling_params, request_id)
    assert finish_reason == "stop" or finish_reason == "length"
    choices = [CompletionResponseChoice(0, text_outputs[0], finish_reason)]
    usage   = UsageInfo(num_prompt_tokens, num_completion_tokens, num_prompt_tokens + num_completion_tokens)
    return CompletionResponse(
      id      = request_id,
      created = create_time_sec,
      model   = request.model,
      choices = choices,
      usage   = usage,
    )

  async def completion_stream_generator(
    self,
    request:    CompletionRequest,
    prompt:     str,
    request_id: str,
    created:    int,
  ) -> AsyncGenerator[str, None]:
    pass
