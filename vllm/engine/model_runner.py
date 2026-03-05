import pickle

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

# from vllm.config import Config
from vllm.models.qwen3 import Qwen3ForCausalLM
# from nanovllm.utils.loader import load_model
# from vllm.layers.sampler import Sampler

class ModelRunner:
  def __init__(
    self, 
    config: Config,  # TODO: Config 类
    rank: int, 
    event: Event | list[Event]
  ):
    hf_config = config.hf_config
    self.config = config
    self.block_size = config.kvcache_block_size
    self.enforce_eager = config.enforce_eager
    self.world_size = config.tensor_parallel_size
    self.rank = rank
    self.event = event
    
    dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
    torch.cuda.set_device(rank)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(hf_config.torch_dtype)
    torch.set_default_device("cuda")

    self.model = Qwen3ForCausalLM(hf_config)
    load_model(self.model, config.model) # TODO: load_model 函数
    self.sampler = Sampler() # TODO: Sampler 类
    self.warmup_model()
    self.allocate_kv_cache()
    if not self.enforce_eager:
      self.capture_cudagraph()

    torch.set_default_device("cpu")
    torch.set_default_dtype(default_dtype)
    
    if self.world_size == 1: 
      return
    
    if rank == 0:
      self.shm = SharedMemory(name="minivllm", create=True, size=2**20)
      dist.barrier()
    else:
      dist.barrier()
      self.shm = SharedMemory(name="minivllm")
      self.loop()

  def write_shm(self, method_name: str, *args):
    assert self.world_size > 1 and self.rank == 0, "write_shm 仅能在 world_size > 1 且 rank == 0 时被调用"
    data                    = pickle.dumps((method_name, *args))
    n                       = len(data)
    self.shm.buf[ : 4]      = n.to_bytes(4, "little")
    self.shm.buf[ : n + 4]  = data
    for event in self.event:
      event.set()

  def read_shm(self):
    assert self.world_size > 1 and self.rank > 0, "read_shm 仅能在 world_size > 1 且 rank > 0 时被调用"
    self.event.wait()
    n                   = int.from_bytes(self.shm.buf[0 : 4], "little")
    method_name, *args  = pickle.loads(self.shm.buf[4 : n + 4])
    self.event.clear()
    return method_name, args
  
  def exit(self):
    if self.world_size > 1:
      self.shm.close()
      dist.barrier()
      if self.rank == 0:
        self.shm.unlink()
    if not self.enforce_eager:
      del self.graphs, self.graph_pool
    torch.cuda.synchronize()
    dist.destroy_process_group()

  def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
      self.write_shm(method_name, *args)
    method = getattr(self, method_name, None)
    if method is None:
      raise ValueError(f"未知方法: {method_name}")
    return method(*args)
  
  def loop(self):
    assert self.world_size > 1 and self.rank != 0, "loop 仅能在 world_size > 1 且 rank > 0 时被调用"
    while True:
      method_name, args = self.read_shm()
      self.call(method_name, *args)
      if method_name == "exit":
        self.exit()
        return  

  def warmup_model(self):
    pass

  def allocate_kv_cache(self):
    pass

  def capture_cudagraph(self):
    pass
