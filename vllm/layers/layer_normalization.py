import torch
from torch import Tensor as tensor, nn

class LayerNormalization(nn.Module):
  def __init__(self, hidden_size_or_gamma: int | tensor, eps: float = 1e-5):
    super().__init__()
    self.weight = nn.Parameter(hidden_size_or_gamma if isinstance(hidden_size_or_gamma, int) else torch.ones(hidden_size_or_gamma))
    self.eps = eps

  @property
  def gamma(self):
    return self.weight
  
  def rms_foward(self, x: tensor) -> tensor:
    pass

  def residual_rms_forward(self, x: tensor, residual: tensor) -> tensor:
    pass

  def foward(self, x: tensor, residual: tensor | None = None) -> tensor:
    pass

if __name__ == "__main__":
  pass