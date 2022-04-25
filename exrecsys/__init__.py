"""
ExRecSYS
========

Provides:
  1. A system recommends items

"""


import torch
import numpy as np

SEED = 434

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True