import torch
from torch import nn, Tensor

class Sampler(nn.Module):
  def __init__(self):
    super().__init__()

  @torch.compile
  def forward(self, logits: Tensor, temperatures: Tensor) -> Tensor:
    logits = logits.float().div_(temperatures.unsqueeze(dim=-1))
    probs = torch.softmax(logits, dim=-1)
    sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    return sample_tokens
