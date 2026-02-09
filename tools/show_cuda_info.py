if __name__ == "__main__":
  import torch
  print(f"PyTorch版本: {torch.__version__}")
  print(f"PyTorch构建的CUDA版本: {torch.version.cuda}")
  print(f"CUDA是否可用: {torch.cuda.is_available()}")
  if torch.cuda.is_available():
      print(f"GPU设备: {torch.cuda.get_device_name(0)}")