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

DATA_DIR = "/nfshomes/asarkar6/trinity/JANe-project/aro/ARO-Visual-Attribution/data/"

class return_aroattribute(Dataset):
    def __init__(self):
        self.df = load_dataset("parquet", data_files=glob.glob(os.path.join(DATA_DIR, "*.parquet")), split="train")
    
    def __getitem__(self, index):
        # get image features
        img_out = self.df[index]['image']
        img_tensor = image_transform(img_out)

        # get text features
        txt = self.df[index]['true_caption']
        txt_tensor = tokenizer(txt)

        neg_txt = self.df[index]['false_caption']
        neg_txt_tensor = tokenizer(neg_txt)

        return img_tensor, txt_tensor, neg_txt_tensor

    def __len__(self):
        return len(self.df)
    







