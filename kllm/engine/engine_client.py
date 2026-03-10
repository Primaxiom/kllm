from typing import TypeAlias, Optional
import multiprocessing as mp
import atexit
import signal
import queue
import threading

import msgspec
import zmq

from kllm.engine.common import EngineStepResult
from kllm.sampling_parameters import SamplingParams
from kllm.config import Config
from kllm.engine.llm_engine import LLMEngine

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
    is_active:  bool                = True
    engine:     Optional[LLMEngine] = None

    def signal_handler(s, f):
      if is_active:
        raise SystemExit()
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT , signal_handler)

    try:
      zmq_ctx = zmq.Context()
      input_queue:  queue.Queue[EngineRequest]  = queue.Queue()
      output_queue: queue.Queue[EngineReply]    = queue.Queue()

      def handle_input_socket():
        input_socket = zmq_ctx.socket(zmq.PULL)
        input_socket.connect(input_path)
        while True:
          frames  = input_socket.recv_multipart(copy=False)
          request = msgspec.msgpack.decode(frames)
          assert isinstance(request, EngineRequest)
          input_queue.put_nowait(request)

      def handle_output_socket():
        output_socket = zmq_ctx.socket(zmq.PUSH)
        output_socket.connect(output_path)
        while True:
          reply   = output_queue.get()
          frames  = msgspec.msgpack.encode(reply)
          output_socket.send_multipart(frames, copy=False)

      input_thread  = threading.Thread(target=handle_input_socket , daemon=True)
      output_thread = threading.Thread(target=handle_output_socket, daemon=True)
      input_thread  .start()
      output_thread .start()

      engine = LLMEngine(**cfg)

      def handle_engine_request(request: EngineRequest):
        if isinstance(request, EngineRequestAdd):
          engine.add_request(
            request.prompt_token_ids, 
            request.sampling_params,
            request.seq_id
          )
        elif isinstance(request, EngineRequestAbort):
          engine.abort_request(request.seq_id)
        else:
          raise ValueError(f"未知的引擎请求: {request}")

      while is_active:
        if not engine.is_finished():
          handle_engine_request(input_queue.get())
        while not input_queue.empty():
          handle_engine_request(input_queue.get_nowait())
        outputs = engine.step()
        if outputs:
          output_queue.put_nowait(EngineReply(outputs=outputs))

    except SystemExit:
      print(f"")
    except Exception as e:
      print(f"引擎异常: {e}")
      if engine is None:
        print(f"引擎启动失败")
    finally:
      is_active = False
      if engine:
        engine.exit()

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
