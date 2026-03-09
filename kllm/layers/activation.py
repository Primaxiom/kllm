import torch
from torch import nn, Tensor
import torch.nn.functional as F

class SiluAndMul(nn.Module):
  '''
Qwen3 MLP 模块中 Silu 和矩阵乘法的结合算子
输入: 门控层和上采样层的输出拼接后的张量, 这两个输出张量的形状是相同的
前向: 将输入平分, 代入公式
输出: 待进行下采样的张量
'''
  def __init__(self):
    super().__init__()

  @torch.compile
  def forward(self, x: Tensor) -> Tensor:
    x, y = x.chunk(2, -1)
    return F.silu(x) * y

class GELUTanhAndMul(nn.Module):
  def __init__(self):
    super().__init__()

  @torch.compile
  def forward(self, x: Tensor) -> Tensor:
    x, y = x.chunk(2, -1)
    return F.gelu(x, approximate="tanh") * y

if __name__ == "__main__":
  layer = SiluAndMul().cuda()
  input_tensor = torch.randn(8, 4000, 8000).cuda()  # Example input tensor
  
  for _ in range(10):  # Warm-up iterations
    _ = layer(input_tensor)

  import time
  times = []
  for _ in range(100):  # Timing iterations
    torch.cuda.synchronize()
    start_time = time.time()
    output_tensor = layer(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()
    times.append(end_time - start_time)
  avg_time = sum(times) / len(times)
  print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
