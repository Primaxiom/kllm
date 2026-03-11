from typing import AsyncGenerator, Optional
import uuid

from transformers import AutoTokenizer, PreTrainedTokenizer
import asyncio

from kllm.config import Config
from kllm.engine.engine_client import EngineClient
from kllm.engine.common import GenerateOutput
from kllm.sampling_parameters import SamplingParams

RequestState = asyncio.Queue[GenerateOutput | None | Exception]

class LLM:
  def __init__(self, cfg: Config):
    self.cfg                                      = cfg
    self.client                                   = EngineClient(cfg)
    self.tokenizer:       PreTrainedTokenizer     = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    self.request_states:  dict[str, RequestState] = {}
    self.output_processor_task                    = asyncio.create_task(self.output_processor())

  def tokenize(self, prompts: list[str]) -> list[list[int]]:
    return self.tokenizer(prompts)["input_ids"]

  def detokenize(self, token_ids: list[list[int]]) -> list[str]:
    return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

  def abort(self, seq_id: str):
    if seq_id in self.request_states:
      self.client.abort_request(seq_id)
      del self.request_states[seq_id]

  async def generate(
    self,
    prompts:          str | list[int],
    sampling_params:  SamplingParams,
    seq_id:           Optional[str] = None,
  ) -> AsyncGenerator[GenerateOutput, None]:
    if seq_id is None:
      seq_id = uuid.uuid4().hex

    try:
      prompt_token_ids: list[int]     = prompts if isinstance(prompts, list) else self.tokenize([prompts])[0]
      request_state:    RequestState  = asyncio.Queue()
      self.request_states[seq_id]     = request_state
      self.client.add_request(
        seq_id,
        prompt_token_ids,
        sampling_params
      )

      while True:
        output = await request_state.get()
        if isinstance(output, Exception):
          raise output
        if output is None:
          break
        yield output
      del self.request_states[seq_id]
      print(f"请求 {seq_id} 生成完毕")

    except (asyncio.CancelledError, GeneratorExit) as e:
      self.abort(seq_id)
      print(f"请求 {seq_id} 生成中止: {e}")
      raise

  async def output_processor(self):
    try:
      while True:
        outputs = await asyncio.to_thread(self.client.get_output)
        for output in outputs:
          seq_id        = output.seq_id
          if seq_id not in self.request_states:
            continue

          new_token_id  = output.new_token_id
          token_str     = self.detokenize([[new_token_id]])[0]
          is_finished   = output.is_finished
          generate_output = GenerateOutput(
            token_str,
            is_finished,
            output.finish_reason,
            output.num_prompt_tokens      if is_finished else 0,
            output.num_completion_tokens  if is_finished else 0,
          )

          request_state = self.request_states[seq_id]
          request_state.put_nowait(generate_output)
          if is_finished:
            request_state.put_nowait(None)
    except Exception as e:
      print(f"output_processor 出错: {e}")
      for request_state in self.request_states.values():
        request_state.put_nowait(e)

  async def exit(self):
    if hasattr(self, "output_processor_task"):
      self.output_processor_task.cancel()
      print(f"output_processor_task 已取消")
    if hasattr(self, "client"):
      await asyncio.to_thread(self.client.exit)
      print(f"client 已退出")
