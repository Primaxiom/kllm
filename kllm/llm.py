

from transformers import AutoTokenizer, PreTrainedTokenizer
import asyncio

from kllm.config import Config
from kllm.engine.engine_client import EngineClient
from kllm.engine.common import GenerateOutput

RequestStates = dict[str, asyncio.Queue[GenerateOutput | None]]

class LLM:
  def __init__(self, cfg: Config):
    self.cfg                            = cfg
    self.client                         = EngineClient(cfg)
    self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    self.request_states: RequestStates  = {}
    self.output_processor_task          = asyncio.create_task(self.output_processor())

  def tokenize(self, prompts: list[str]) -> list[list[int]]:
    return self.tokenizer(prompts)["input_ids"]

  def detokenize(self, token_ids: list[list[int]]) -> list[str]:
    return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

  def abort(self, seq: str):
    if seq in self.request_states:
      self.client.abort_request(seq)
      del self.request_states[seq]

  async def generate(self):
    pass

  async def output_processor(self):
    pass

  async def exit(self):
    self.output_processor_task.cancel()
    await asyncio.to_thread(self.client.exit)
