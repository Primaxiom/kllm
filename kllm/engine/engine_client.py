from typing import TypeAlias

import msgpack
import msgspec

from kllm.engine.common import EngineStepResult
from kllm.sampling_parameters import SamplingParams

class EngineRequestBase(
  msgspec.Struct,
  array_like    = True,
  omit_defaults = True,
  gc            = False,
  tag           = True,
):
  pass

class EngineRequestAdd(EngineRequestBase):
  seq_id:           int
  prompt_token_ids: list[int]
  sampling_params:  SamplingParams

class EngineRequestAbort(EngineRequestBase):
  seq_id:           int

EngineRequest: TypeAlias = EngineRequestAdd | EngineRequestAbort

class EngineReply(
  msgspec.Struct,
  array_like    = True,
  omit_defaults = True,
  gc            = False,
):
  outputs: list[EngineStepResult]
  