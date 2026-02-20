import torch
from torch import Tensor, nn

class LayerNormalization(nn.Module):
  '''
RMS Layer Normalization
一个带有残差功能的均方根归一化层
RMSNorm(x) = (x / sqrt(mean(x²) + ε)) ⊙ γ
'''

  def __init__(self, hidden_size_or_gamma: int | Tensor, eps: float = 1e-5):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size_or_gamma) if isinstance(hidden_size_or_gamma, int) else hidden_size_or_gamma)
    self.eps = eps

  @property
  def gamma(self):
    return self.weight
  
  def rms_forward(self, x: Tensor) -> Tensor:
    # RMSNorm(x) = (x / sqrt(mean(x²) + ε)) ⊙ γ
    variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
    return x * torch.rsqrt(variance) * self.weight

  def residual_rms_forward(self, x: Tensor, residual: Tensor) -> Tensor:
    x += residual
    return self.rms_forward(x), x

  def forward(self, x: Tensor, residual: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
    return self.rms_forward(x) if residual is None else self.residual_rms_forward(x, residual)

if __name__ == "__main__":
  # Example usage
  x = torch.randn(8,4000,8000).cuda()
  gamma = torch.full((8000,), 0.5, device="cuda", dtype=x.dtype)
  layer = LayerNormalization(hidden_size_or_gamma=gamma).cuda()
  residual = torch.full_like(x,fill_value=1)

  for _ in range(10): # Warm-up iterations
    _ = layer(x)
  
  import time
  # Without residuals
  times = [] 
  for _ in range(100): # Timing iterations
    torch.cuda.synchronize()
    start_time = time.time()
    _ = layer(x)
    torch.cuda.synchronize()
    end_time = time.time()
    times.append(end_time - start_time)
  avg_time = sum(times) / len(times)
  print(f"[Without residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")

  # With residuals
  times.clear()
  for _ in range(100): # Timing iterations
    torch.cuda.synchronize()
    start_time = time.time()
    _ = layer(x,residual)
    torch.cuda.synchronize()
    end_time = time.time()
    times.append(end_time - start_time)
  avg_time = sum(times) / len(times)
  print(f"[With residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")