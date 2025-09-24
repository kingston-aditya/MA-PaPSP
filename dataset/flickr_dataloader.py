from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import torchvision.transforms as v2

import pandas as pd
import os
from PIL import Image
import pdb 

DATA_DIR = "/nfshomes/asarkar6/trinity/JANe-project/flickr/flickr30k"

class return_flickr(Dataset):
    def __init__(self):
        self.df = pd.read_csv(os.path.join(DATA_DIR, "flickr_annotations_30k.csv"), dtype=object)
    
    def __getitem__(self, index):
        # get image features
        img_pth = os.path.join(DATA_DIR, "flickr30k-images", self.df["filename"][index])
        img_out = Image.open(img_pth).convert('RGB')

        # get text features
        txt = self.df["raw"][index]

        return {"prompts": txt, "images": img_out}

    def __len__(self):
        return len(self.df)
    
def collate_fn_cap(batch):
        prompts = [item["prompts"] for item in batch]
        images = [item["images"] for item in batch]
        return {
            "prompts": prompts,
            "images": images
        }

# if __name__ == "__main__":
#     dataset = return_flickr()
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_cap, num_workers=4)

#     for i, batch in enumerate(dataloader):
#         pdb.set_trace()
#         print(i, batch["prompts"], batch["images"])
#         if i == 2:
#             break






