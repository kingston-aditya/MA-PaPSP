from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as v2
from tokenizer import SimpleTokenizer
import glob
from datasets import load_dataset

import json
import os
from PIL import Image
# import pdb 

tokenizer = SimpleTokenizer()

image_transform = v2.Compose(
    [
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize([0.5], [0.5]),
    ]
)

DATA_DIR = "/nfshomes/asarkar6/trinity/JANe-project/whats_up_vlms/"

class return_whatsup(Dataset):
    def __init__(self):
        self.df = load_dataset("json", data_files=os.path.join(DATA_DIR, "examples.jsonl"), split="train")
    
    def __getitem__(self, index):
        # get image features
        img_out = Image.open(os.path.join(DATA_DIR, "images", self.df[index]['image_file_name']))
        img_tensor = image_transform(img_out)

        # get text features
        txt = self.df[index]['positive_caption'][0]
        txt_tensor = tokenizer(txt)

        neg_txt = self.df[index]['negative_caption'][0]
        neg_txt_tensor = tokenizer(neg_txt)

        return img_tensor, txt_tensor, neg_txt_tensor

    def __len__(self):
        return len(self.df)






