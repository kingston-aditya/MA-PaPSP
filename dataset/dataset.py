import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
from ..configs.config import get_config
from ..utils.run_retrieval import find_retrieved_items

class COCO_CC12M(Dataset):
    def __init__(self, k):
        super().__init__()
        config = get_config()
        self.k = k
        self.X = torch.Tensor(np.load(config["coco_train_data"]))
        self.Y = torch.Tensor(np.load(config["coco_label_data"]))
        self.Xr, self.Yr = find_retrieved_items(self.X, self.k)
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        out = {
            "image_embed": self.X[index],
            "text_embed": self.Y[index],
            "ret_image_embed": self.Xr[index],
            "ret_text_embed": self.Yr[index]
        }
        return out

        

