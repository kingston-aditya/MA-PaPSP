import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import sys
sys.path.insert(1, "/data/aditya/JANe/")
from configs.config import get_config
sys.path.insert(1, "/data/aditya/JANe/utils")
from run_retrieval import find_retrieved_items

config = get_config()
    
class COCO_CC12M(Dataset):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.X = torch.Tensor(np.load(config["coco_train_data"]))
        self.Y = torch.Tensor(np.load(config["coco_label_data"]))
        self.Xr, self.Yr = find_retrieved_items().get_array(self.X, self.k)
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        out = {
            "input": self.X[index],
            "output": self.Y[index],
            "ret_input": self.Xr[index],
            "ret_output": self.Yr[index]
        }
        return out

        

