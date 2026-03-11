import time
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

class ErrorResponse(BaseModel):
  object: str = "error"
  message: str
  type: str
  param: Optional[str] = None
  code: int

class CompletionRequest(BaseModel):
  model: str
  prompt: Union[str, List[str]]
  max_tokens: Optional[int] = None
  temperature: Optional[float] = 1.0
  stream: Optional[bool] = False
  ignore_eos: Optional[bool] = False

class UsageInfo(BaseModel):
  prompt_tokens: int = 0
  total_tokens: int = 0
  completion_tokens: Optional[int] = 0

class CompletionResponseChoice(BaseModel):
  index: int
  text: str
  finish_reason: Optional[Literal["stop", "length"]] = None

class CompletionResponse(BaseModel):
  id: str = Field(default_factory=lambda: f"cmpl-{int(time.time())}")
  object: str = "text_completion"
  created: int = Field(default_factory=lambda: int(time.time()))
  model: str
  choices: List[CompletionResponseChoice]
  usage: UsageInfo

class CompletionResponseStreamChoice(BaseModel):
  index: int
  text: str
  finish_reason: Optional[Literal["stop", "length"]] = None

class CompletionStreamResponse(BaseModel):
  id: str = Field(default_factory=lambda: f"cmpl-{int(time.time())}")
  object: str = "text_completion"
  created: int = Field(default_factory=lambda: int(time.time()))
  model: str
  choices: List[CompletionResponseStreamChoice]
  usage: Optional[UsageInfo] = None
