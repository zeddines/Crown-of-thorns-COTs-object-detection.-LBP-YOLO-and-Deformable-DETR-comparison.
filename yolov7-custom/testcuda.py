import torch


import torch

torch.cuda.empty_cache()

from GPUtil import showUtilization as gpu_usage
print(gpu_usage())