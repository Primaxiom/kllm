import torch
from torch import nn, Tensor
import torch.nn.functional as F

class SiluAndMul(nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, x: Tensor) -> Tensor:
    x, y = x.chunk(2, -1)
    return F.silu(x) * y

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