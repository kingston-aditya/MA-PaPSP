from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import torchvision.transforms as v2

import json
import os
from PIL import Image
import pdb 

DATA_DIR = "/fs/cml-datasets/coco/"

class return_coco(Dataset):
    def __init__(self):
        f = open(os.path.join(DATA_DIR, "annotations", "captions_val2017.json"), "r")
        self.json_obj = json.load(f)
        f.close()
    
    def __getitem__(self, index):
        # get text features
        txt = self.json_obj["annotations"][index]["caption"]

        # get paths
        num = f"{self.json_obj['annotations'][index]['image_id']:012d}"
        img_pth = os.path.join(DATA_DIR, "images/val2017", f"{num}.jpg")
        img_out = Image.open(img_pth).convert('RGB')

        # get image ids
        tag = self.json_obj['annotations'][index]['image_id']

        return {"prompts": txt, "images": img_out, "tags": tag}

    def __len__(self):
        return len(self.json_obj["annotations"])
    
# def collate_fn_cap(batch):
#         prompts = [item["prompts"] for item in batch]
#         images = [item["images"] for item in batch]
#         return {
#             "prompts": prompts,
#             "images": images
#         }

# if __name__ == "__main__":
#     dataset = return_coco()
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_cap, num_workers=4)

#     for i, batch in enumerate(dataloader):
#         pdb.set_trace()
#         print(i, batch["prompts"], batch["images"])
#         if i == 2:
#             break






