import os
import torch
import numpy as np

torch.manual_seed(20101014)
np.random.seed(20191001)

USE_MULTIPLE_GPUS = True

NUMBER_OF_CLASSES = 10

if USE_MULTIPLE_GPUS:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
    DEVICE = torch.device("cuda")
    BATCH_SIZE = 350
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1