import math
import pickle

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from vllm.config import Config
from vllm.models.qwen3 import Qwen3ForCausalLM
from vllm.utils.loader import load_model
from vllm.layers.sampler import Sampler
from vllm.engine.sequence import Sequence

class ModelRunner:
  def __init__(
    self, 
    config: Config,
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
    load_model(self.model, config.model)
    self.sampler = Sampler()
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

  def run(self):
    pass

  def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_num_batched_tokens  =     self.config.max_num_batched_tokens
    max_model_len           =     self.config.max_model_len
    num_seqs                = min(self.config.max_num_seqs, max_num_batched_tokens // max_model_len)
    seqs                    = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
    self.run(seqs, True)
    torch.cuda.empty_cache()

  def allocate_kv_cache(self):
    config                  = self.config
    hf_config               = config.hf_config

    free_mem, total_mem     = torch.cuda.mem_get_info()
    used_mem                = total_mem - free_mem
    total_mem              *= config.gpu_memory_utilization
    torch_peak_used_mem     = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    torch_current_used_mem  = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    available_mem           = total_mem - used_mem - (torch_peak_used_mem - torch_current_used_mem)

    kv_cache_shape          = [
      2,
      hf_config.num_hidden_layers,
      1,
      self.block_size,
      hf_config.num_key_value_heads // self.world_size,
      getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    ]

    config.num_kvcache_blocks = int(available_mem) // (math.prod(kv_cache_shape) * hf_config.torch_dtype.itemsize)
    assert config.num_kvcache_blocks >= 1, f"rank {self.rank} 上的显存空间少于 1 个 KVCache 块所需的空间"
    kv_cache_shape[2]         = config.num_kvcache_blocks
    self.kv_cache             = torch.empty(*kv_cache_shape)

    layer_id = 0
    for module in self.model.modules():
      if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id      += 1


  def capture_cudagraph(self):
    pass
