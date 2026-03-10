

from transformers import AutoTokenizer
import asyncio

from kllm.config import Config
from kllm.engine.engine_client import EngineClient
from kllm.engine.common import GenerateOutput

RequestStates = dict[str, asyncio.Queue[GenerateOutput | None]]

class LLM:
  def __init__(self, cfg: Config):
    self.cfg                            = cfg
    self.client                         = EngineClient(cfg)
    self.tokenizer                      = AutoTokenizer.from_pretrained(cfg.model)
    self.request_states: RequestStates  = {}
    self.output_processor_task          = asyncio.create_task(self.output_processor())

  def tokenize(self):
    pass

  def detokenize(self):
    pass

  def abort(self):
    pass

  async def generate(self):
    pass

  async def output_processor(self):
    pass

  async def exit(self):
    pass
