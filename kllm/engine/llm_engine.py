
from dataclasses import fields
from multiprocessing.synchronize import Event
from multiprocessing.context import SpawnProcess
import atexit
from time import perf_counter
from typing import Optional

import torch.multiprocessing as mp
from transformers import AutoTokenizer
from tqdm import tqdm

from kllm.config import Config
from kllm.engine.model_runner import ModelRunner
from kllm.engine.scheduler import Scheduler
from kllm.engine.sequence import Sequence
from kllm.sampling_parameters import SamplingParams
from kllm.engine.common import EngineStepResult

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
    if hasattr(self, "model_runner"):
      self.model_runner.call("exit")
      del self.model_runner
    if hasattr(self, "processes"):
      for process in self.processes:
        process.join()

  def add_request(
    self, 
    prompt:           str | list[int], 
    sampling_params:  SamplingParams,
    seq_id:           Optional[str] = None,
  ) -> str:
    token_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
    seq = Sequence(token_ids, sampling_params, seq_id)
    self.scheduler.add_seq(seq)
    return seq.seq_id

  def abort_request(self, seq_id: str):
    seq = self.scheduler.get_seq(seq_id)
    if seq:
      seq.finish_reason = "abort"
      self.scheduler.finish_seq(seq)

  def is_finished(self):
    return self.scheduler.is_finished()

  def step(self) -> list[EngineStepResult]:
    seqs, is_prefill  = self.scheduler.schedule()
    token_ids         = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids)
    outputs           = [
      EngineStepResult(
        seq_id                = seq.seq_id,
        new_token_id          = seq.last_token,
        is_finished           = seq.is_finished,
        finish_reason         = seq.finish_reason,
        num_prompt_tokens     = seq.num_prompt_tokens,
        num_completion_tokens = seq.num_completion_tokens,
      ) for seq in seqs
    ]
    return outputs

  def generate(
    self,
    prompts:          list[str]       | list[list[int]],
    sampling_params:  SamplingParams  | list[SamplingParams],
    use_tqdm: bool = True,
  ) -> list[dict[str, any]]:
    seq_ids = []
    if use_tqdm:
      process_bar = tqdm(total=len(prompts), desc="生成中", dynamic_ncols=True)
    if not isinstance(sampling_params, list):
      sampling_params = [sampling_params] * len(prompts)
    for prompt, params in zip(prompts, sampling_params):
      seq_id = self.add_request(prompt, params)
      seq_ids.append(seq_id)
    
    outputs: dict[str, list[int]] = {}
    throughput: list[float]       = [.0, .0]
    while not self.is_finished():
      start = perf_counter()
      step_outputs = self.step()
      if use_tqdm:
        end = perf_counter()
        is_prefill = step_outputs[0].num_completion_tokens <= 1
        num_tokens = sum(o.num_prompt_tokens for o in step_outputs) if is_prefill else len(step_outputs)
        throughput[is_prefill] = num_tokens / (end - start)
        process_bar.set_postfix({
            "Prefill": f"{int(throughput[1])}tok/s",
            "Decode" : f"{int(throughput[0])}tok/s",
        })

      for step_result in step_outputs:
        seq_id, token_id = step_result.seq_id, step_result.new_token_id
        if seq_id not in outputs:
          outputs[seq_id] = []
        outputs[seq_id].append(token_id)
        if use_tqdm:
          process_bar.update(1)

    outputs = [outputs[seq_id] for seq_id in seq_ids]
    outputs = [
      {
        "text": self.tokenizer.decode(token_ids),
        "token_ids": token_ids,
      }
      for token_ids in outputs
    ]
    if use_tqdm:
      process_bar.close()
    return outputs
