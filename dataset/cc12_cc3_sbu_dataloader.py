from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from tokenizer import SimpleTokenizer

from datasets import load_dataset
import json
import os
# import pdb 

tokenizer = SimpleTokenizer()

DATA_DIR = "/nfshomes/asarkar6/trinity/JANe-project/ccs_data/"

class return_cc12_cc3_sbu(Dataset):
    def __init__(self):
        f = open(os.path.join(DATA_DIR, "ccs_filtered.json"))
        self.json_obj = json.load(f)
        f.close()
    
    def __getitem__(self, index):
        out = self.json_obj[index]["caption"]
        tokens_txt = tokenizer(out)
        return torch.Tensor(tokens_txt)

    def __len__(self):
        return len(self.json_obj)






