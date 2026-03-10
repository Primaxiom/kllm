from typing import TypeAlias
import multiprocessing as mp
import atexit

import msgspec
import zmq

from kllm.engine.common import EngineStepResult
from kllm.sampling_parameters import SamplingParams
from kllm.config import Config

class EngineRequestBase(
  msgspec.Struct,
  array_like    = True,
  omit_defaults = True,
  gc            = False,
  tag           = True,
):
  pass

class EngineRequestAdd(EngineRequestBase):
  seq_id:           str
  prompt_token_ids: list[int]
  sampling_params:  SamplingParams

class EngineRequestAbort(EngineRequestBase):
  seq_id:           str

EngineRequest: TypeAlias = EngineRequestAdd | EngineRequestAbort

class EngineReply(
  msgspec.Struct,
  array_like    = True,
  omit_defaults = True,
  gc            = False,
):
  outputs: list[EngineStepResult]

class EngineClient:
  def __init__(self, cfg: Config):
    self.config = cfg
    self.mp_ctx = mp.get_context("spawn")
    self.zmq_ctx = zmq.Context()

    self.input_path = "tcp://127.0.0.1:6666"
    self.output_path = "tcp://127.0.0.1:6667"

    self.input_socket = self.zmq_ctx.socket(zmq.PUSH)
    self.input_socket.bind(self.input_path)

    self.output_socket = self.zmq_ctx.socket(zmq.PULL)
    self.output_socket.bind(self.output_path)

    self.engine_process = self.mp_ctx.Process(
      target=self.run_engine_loop,
      name="kllm_engine",
      args=(
        self.config,
        self.input_path,
        self.output_path,
      ),
    )
    self.engine_process.start()

    self.is_active = True
    self.poller = zmq.Poller()
    self.poller.register(self.output_socket, zmq.POLLIN)

    atexit.register(self.exit)

  @staticmethod
  def run_engine_loop(
    cfg: Config,
    input_path: str,
    output_path: str,
  ):
    pass

  def exit(self):
    self.is_active = False
    p = self.engine_process
    p.terminate()
    p.join()
    self.input_socket.close()

  def add_request(
    self,
    seq_id:           str,
    prompt_token_ids: list[int],
    sampling_params:  SamplingParams,
  ):
    if self.is_active:
      frames = msgspec.msgpack.encode(
        EngineRequestAdd(
          seq_id            = seq_id,
          prompt_token_ids  = prompt_token_ids,
          sampling_params   = sampling_params,
        )
      )
      self.input_socket.send_multipart(frames, copy=False)

  def abort_request(self, seq_id):
    if self.is_active:
      frames = msgspec.msgpack.encode(EngineRequestAbort(seq_id=seq_id))
      self.input_socket.send_multipart(frames, copy=False)

  def get_output(self) -> list[EngineStepResult]:
    while self.is_active:
      sockets = dict(self.poller.poll(timeout=1000))
      if self.output_socket in sockets:
        frames  = self.output_socket.recv_multipart(flags=zmq.DONTWAIT, copy=False)
        reply   = msgspec.msgpack.decode(frames, type=EngineReply)
        assert isinstance(reply, EngineReply)
        return reply.outputs
