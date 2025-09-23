from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as v2
from tokenizer import SimpleTokenizer
import glob
from datasets import load_dataset

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

DATA_DIR = "/nfshomes/asarkar6/trinity/JANe-project/foil/"

class return_foil(Dataset):
    def __init__(self):
        self.df = load_dataset("json", data_files=os.path.join(DATA_DIR, "foilv1.0_test_2017.json"), split="train")
    
    def __getitem__(self, index):
        # get image features
        img_out = Image.open(os.path.join(DATA_DIR, "coco_images", f"COCO_val2014_000000{self.df["annotations"][index]['image_id']}.jpg"))
        img_tensor = image_transform(img_out)

        # get text features
        if index % 2 == 0:
            txt = self.df["annotations"][index]['caption']
            txt_tensor = tokenizer(txt)

            neg_txt = self.df["annotations"][index+1]['caption']
            neg_txt_tensor = tokenizer(neg_txt)
        else:
            txt = self.df["annotations"][index]['caption']
            txt_tensor = tokenizer(txt)

            neg_txt = self.df["annotations"][index-1]['caption']
            neg_txt_tensor = tokenizer(neg_txt)

        return img_tensor, txt_tensor, neg_txt_tensor

    def __len__(self):
        return len(self.df)






