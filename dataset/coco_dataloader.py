from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from tokenizer import SimpleTokenizer
import torchvision.transforms as v2

import json
import os
from PIL import Image
# import pdb 

tokenizer = SimpleTokenizer()

DATA_DIR = "/fs/cml-datasets/coco/"

image_transform = v2.Compose(
    [
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize([0.5], [0.5]),
    ]
)

class return_coco(Dataset):
    def __init__(self):
        f = open(os.path.join(DATA_DIR, "annotations", "captions_val2017.json"), "r")
        self.json_obj = json.load(f)
        f.close()
    
    def __getitem__(self, index):
        # get text features
        txt = self.json_obj["annotations"][index]["caption"]
        txt_tensor = tokenizer(txt)

        # get paths
        img_pth = os.path.join(DATA_DIR, "images", f"00000{self.json_obj["annotations"][index]["image_id"]}.jpg")

        # get image features
        img_out = Image.open(img_pth).convert('RGB')
        img_tensor = image_transform(img_out)

        return img_tensor, txt_tensor

    def __len__(self):
        return len(self.json_obj["annotations"])






