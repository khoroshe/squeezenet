import os
import torch
import numpy as np

torch.manual_seed(20101014)
np.random.seed(20191001)

USE_MULTIPLE_GPUS = True

if USE_MULTIPLE_GPUS:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
    DEVICE = torch.device("cuda")
    BATCH_SIZE = 200
    WORKERS = 8
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1
    WORKERS = 2