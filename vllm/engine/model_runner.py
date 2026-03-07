import math
import pickle

import torch
import torch.distributed as dist
from torch import cuda, Tensor
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from vllm.config import Config
from vllm.models.qwen3 import Qwen3ForCausalLM
from vllm.utils.loader import load_model
from vllm.layers.sampler import Sampler
from vllm.engine.sequence import Sequence
from vllm.utils.context import set_context, reset_context, get_context

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
    
    dist.init_process_group("gloo", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
    cuda.set_device(rank)
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
    cuda.synchronize()
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

  def prepare_block_tables(self, seqs: list[Sequence]) -> Tensor:
    max_len       = max(len(seq.block_table) for seq in seqs)
    block_tables  = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
    block_tables  = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    return block_tables

  def prepare_prefill(self, seqs: list[Sequence]) -> tuple[Tensor, Tensor]:
    input_ids     = []
    positions     = []
    cu_seqlens_q  = [0]
    cu_seqlens_k  = [0]
    max_seqlen_q  = 0
    max_seqlen_k  = 0
    slot_mapping  = []

    for seq in seqs:
      seqlen        = len(seq)
      seqlen_q      = seqlen - seq.num_cached_tokens
      seqlen_k      = seqlen
      input_ids   .extend(seq.token_ids[seq.num_cached_tokens : seqlen])
      positions   .extend(        range(seq.num_cached_tokens , seqlen))
      cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
      cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
      max_seqlen_q  = max(max_seqlen_q, seqlen_q)
      max_seqlen_k  = max(max_seqlen_k, seqlen_k)

      if not seq.block_table: 
        continue
      for logical_block_id in range(seq.num_cached_blocks, seq.num_blocks):
        start = seq.block_table[logical_block_id] * self.block_size
        end   = start + (self.block_size if logical_block_id != seq.num_blocks - 1 else seq.last_block_num_tokens)
        slot_mapping.extend(range(start, end))

    input_ids     = torch.tensor(input_ids,     dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions     = torch.tensor(positions,     dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_q  = torch.tensor(cu_seqlens_q,  dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_k  = torch.tensor(cu_seqlens_k,  dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    slot_mapping  = torch.tensor(slot_mapping,  dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    block_tables  = self.prepare_block_tables(seqs) if cu_seqlens_q[-1] < cu_seqlens_k[-1] else None
    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
    return input_ids, positions

  def prepare_decode(self, seqs: list[Sequence]) -> tuple[Tensor, Tensor]:
    input_ids     = []
    positions     = []
    slot_mapping  = []
    context_lens  = []

    for seq in seqs:
      input_ids   .append(seq.last_token)
      positions   .append(len(seq) - 1)
      context_lens.append(len(seq))
      slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

    input_ids     = torch.tensor(input_ids,     dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions     = torch.tensor(positions,     dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    slot_mapping  = torch.tensor(slot_mapping,  dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    context_lens  = torch.tensor(context_lens,  dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    block_tables  = self.prepare_block_tables(seqs)
    set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
    return input_ids, positions

  def prepare_sample(self, seqs: list[Sequence]) -> Tensor:
    temperatures = [seq.temperature for seq in seqs]
    temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
    return temperatures

  @torch.inference_mode()
  def run_model(
    self, 
    input_ids: Tensor, 
    positions: Tensor, 
    is_prefill: bool
  ) -> Tensor:
    batch_size  = input_ids.size(0)
    if is_prefill or self.enforce_eager or batch_size > 512:
      return self.model.compute_logits(self.model(input_ids, positions))
    context     = get_context()
    graph       = self.graphs[next(bs for bs in self.graph_batch_size if bs >= batch_size)]
    graph_vars  = self.graph_vars
    graph_vars["input_ids"]   [ : batch_size] = input_ids
    graph_vars["positions"]   [ : batch_size] = positions
    graph_vars["slot_mapping"].fill_(-1)
    graph_vars["slot_mapping"][ : batch_size] = context.slot_mapping
    graph_vars["context_lens"].zero_()
    graph_vars["context_lens"][ : batch_size] = context.context_lens
    graph_vars["block_tables"][ : batch_size, : context.block_tables.size(1)] = context.block_tables
    graph.replay()
    return self.model.compute_logits(graph_vars["outputs"][ : batch_size])

  def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    input_ids, positions  = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    logits                = self.run_model(input_ids, positions, is_prefill)
    token_ids             = None
    if self.rank == 0:
      temperatures  = self.prepare_sample(seqs)
      token_ids     = self.sampler(logits, temperatures).tolist()
    reset_context()
    return token_ids

  def warmup_model(self):
    cuda.empty_cache()
    cuda.reset_peak_memory_stats()
    max_num_batched_tokens  =     self.config.max_num_batched_tokens
    max_model_len           =     self.config.max_model_len
    num_seqs                = min(self.config.max_num_seqs, max_num_batched_tokens // max_model_len)
    seqs                    = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
    self.run(seqs, True)
    cuda.empty_cache()

  def allocate_kv_cache(self):
    config                  = self.config
    hf_config               = config.hf_config

    free_mem, total_mem     = cuda.mem_get_info()
    used_mem                = total_mem - free_mem
    total_mem              *= config.gpu_memory_utilization
    torch_peak_used_mem     = cuda.memory_stats()["allocated_bytes.all.peak"]
    torch_current_used_mem  = cuda.memory_stats()["allocated_bytes.all.current"]
    available_mem           = total_mem - used_mem - (torch_peak_used_mem - torch_current_used_mem)

    kv_cache_shape          = [
      2,
      hf_config.num_hidden_layers,
      1,
      self.block_size,
      hf_config.num_key_value_heads // self.world_size,
      getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    ]

    config.num_kvcache_blocks = int(available_mem) // (math.prod(kv_cache_shape) * hf_config.dtype.itemsize)
    assert config.num_kvcache_blocks >= 1, f"rank {self.rank} 上的显存空间少于 1 个 KVCache 块所需的空间"
    kv_cache_shape[2]         = config.num_kvcache_blocks
    self.kv_cache             = torch.empty(*kv_cache_shape)

    layer_id = 0
    for module in self.model.modules():
      if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id      += 1

  @torch.inference_mode()
  def capture_cudagraph(self):
    config          = self.config
    hf_config       = config.hf_config
    max_batch_size  = min(config.max_num_seqs, 512)
    max_num_blocks  = (config.max_model_len + self.block_size - 1) // self.block_size

    input_ids       = torch.zeros(max_batch_size, dtype=torch.int64)
    positions       = torch.zeros(max_batch_size, dtype=torch.int64)
    slot_mapping    = torch.zeros(max_batch_size, dtype=torch.int32)
    context_lens    = torch.zeros(max_batch_size, dtype=torch.int32)
    block_tables    = torch.zeros(max_batch_size, max_num_blocks, dtype=torch.int32)
    outputs         = torch.zeros(max_batch_size, hf_config.hidden_size)

    self.graph_batch_size = [1, 2, 4, 8] + list(range(16, max_batch_size + 1, 16))
    self.graphs           = {}
    self.graph_pool       = None

    for batch_size in reversed(self.graph_batch_size):
      graph = cuda.CUDAGraph()
      set_context(
        False, 
        slot_mapping=slot_mapping[ : batch_size], 
        context_lens=context_lens[ : batch_size], 
        block_tables=block_tables[ : batch_size], 
      )

      outputs[ : batch_size] = self.model(input_ids[ : batch_size], positions[ : batch_size])
      with cuda.graph(graph, self.graph_pool):
        outputs[ : batch_size] = self.model(input_ids[ : batch_size], positions[ : batch_size])
      if self.graph_pool is None:
        self.graph_pool = graph.pool()
      self.graphs[batch_size] = graph

      cuda.synchronize()
      reset_context()

    self.graph_vars = dict(
      input_ids     = input_ids,
      positions     = positions,
      slot_mapping  = slot_mapping,
      context_lens  = context_lens,
      block_tables  = block_tables,
      outputs       = outputs,
    )
