from typing import Optional
from dataclasses import dataclass

@dataclass
class EngineStepResult:
  seq_id:                 str
  new_token_id:           int
  is_finished:            bool
  finish_reason:          Optional[str]
  num_prompt_tokens:      int
  num_completion_tokens:  int
