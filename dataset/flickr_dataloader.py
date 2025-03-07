from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from tokenizer import SimpleTokenizer
from config import get_config
config = get_config()
import json
import os
from PIL import Image
# import pdb 

tokenizer = SimpleTokenizer()

class return_flickr(Dataset):
    def __init__(self):
        f = open(os.path.join(config["data_dir"], "ccs_filtered.json"))
        self.json_obj = json.load(f)
        f.close()
    
    def __getitem__(self, index):
        out = self.json_obj[index]["caption"]
        tokens_txt = tokenizer(out)
        return torch.Tensor(tokens_txt)

    def __len__(self):
        return len(self.json_obj)






