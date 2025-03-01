import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
from ..configs.config import get_config
from ..utils.run_retrieval import run_retrieval

class COCO_CC12M(Dataset):
    def __init__(self, k):
        super().__init__()
        config = get_config()
        self.k = k
        self.X = np.load(config["coco_train_data"])
        self.Y = np.load(config["coco_label_data"])
        self.Xr, self.Yr = run_retrieval(config["cc12m_folder"]).retrieve_X(self.X, self.k)
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        out = {

        }
        return super().__getitem__(index)

        

