import os, sys
sys.path.insert(0, os.getcwd())
import argparse
from contextlib import asynccontextmanager
import gc

from fastapi import FastAPI, Request
import uvicorn

from kllm.llm import LLM
from kllm.config import Config
from kllm.entrypoints.serving import OpenAIServingCompletion
from kllm.entrypoints.protocol import CompletionRequest

@asynccontextmanager
async def lifespan(app: FastAPI):
  llm = None

  try:
    model = args.model
    llm   = LLM(
      Config(
        model         = os.path.expanduser(f"~/{model}/"),
        max_model_len = args.context_len,
      )
    )

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

@app.post("/v0/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
  return await raw_request \
              .app \
              .state \
              .serving_completion \
              .create_completion(request)

def run_server(args: argparse.Namespace):
  uvicorn.run(
    app,
    host=args.host,
    port=args.port,
  )

if __name__ == "__main__":
  os.environ["USE_LIBUV"] = "0"
  parser  = argparse.ArgumentParser(description="kLLM!")
  parser  .add_argument("--host",         type=str, default="localhost",  help="Host name")
  parser  .add_argument("--port",         type=int, default=8000,         help="Port number")
  parser  .add_argument("--model",        type=str, required=True,        help="Model name")
  parser  .add_argument("--context-len",  type=int, default=4096,         help="Max context length of the model")
  parser  .add_argument("--enforce-eager",          action="store_true",  help="Enforce eager execution, disable CUDA graph")
  args    = parser.parse_args()
  run_server(args)
