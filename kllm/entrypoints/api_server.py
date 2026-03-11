import argparse
from contextlib import asynccontextmanager
import gc

from fastapi import FastAPI, Request

from kllm.llm import LLM
from kllm.config import Config
from kllm.entrypoints.serving import OpenAIServingCompletion
from kllm.entrypoints.protocol import CompletionRequest

@asynccontextmanager
async def lifespan(app: FastAPI):
  llm = None

  try:
    model = args.model
    llm   = LLM(Config(model=model))

    app.state.llm                 = llm
    app.state.model_name          = model
    app.state.serving_completion  = OpenAIServingCompletion(llm, model)

    yield

  finally:
    if hasattr(app.state, "model_name"):
      del app.state.model_name
    if hasattr(app.state, "serving_completion"):
      del app.state.serving_completion
    if llm is not None:
      await llm.exit()
      del llm
    gc.collect()
    

app = FastAPI(lifespan=lifespan)

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
  return await raw_request \
              .state \
              .serving_completion \
              .create_completion(request)

if __name__ == "__main__":
  parser  = argparse.ArgumentParser(description="kLLM!")
  args    = parser.parse_args()
