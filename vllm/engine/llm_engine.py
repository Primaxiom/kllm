
from dataclasses import fields
from multiprocessing.synchronize import Event
from multiprocessing.context import SpawnProcess
import atexit

import torch.multiprocessing as mp
from transformers import AutoTokenizer

from vllm.config import Config
from vllm.engine.model_runner import ModelRunner
from vllm.engine.scheduler import Scheduler
from vllm.engine.sequence import Sequence
from vllm.sampling_parameters import SamplingParams

class LLMEngine:
  def __init__(self, model: str, **kwargs):
    cfg_fields  = {field.name for field in fields(Config)}
    cfg_kwargs  = {k: v for k, v in kwargs.items() if k in cfg_fields}
    cfg         = Config(model, **cfg_kwargs)

    self.events:    list[Event]         = []
    self.processes: list[SpawnProcess]  = []
    ctx                                 = mp.get_context("spawn")
    for i in range(1, cfg.tensor_parallel_size):
      event    = ctx.Event()
      process  = ctx.Process(target=ModelRunner, args=(cfg, i, event))
      process       .start()
      self.events   .append(event)
      self.processes.append(process)

    self.model_runner = ModelRunner(cfg, 0, self.events)
    self.tokenizer    = AutoTokenizer.from_pretrained(cfg.model, use_fast=True)
    cfg.eos           = self.tokenizer.eos_token_id
    self.scheduler    = Scheduler(cfg)
    atexit.register(self.exit)

  def exit(self):
    self.model_runner.call("exit")
    del self.model_runner
    for process in self.processes:
      process.join()

  def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
    token_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
    seq = Sequence(token_ids, sampling_params)
    self.scheduler.add_seq(seq)

  def is_finished(self):
    return self.scheduler.is_finished()

  def step(self):
    pass

  def generate(
    self,
    prompts:          list[str]       | list[list[int]],
    sampling_params:  SamplingParams  | list[SamplingParams],
  ) -> list[str]:
    pass
