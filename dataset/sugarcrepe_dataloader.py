from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as v2
from tokenizer import SimpleTokenizer
import glob

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

DATA_DIR = "/nfshomes/asarkar6/trinity/JANe-project/sugar-crepe/data/*.json"
IMG_DIR = "/fs/cml-datasets/coco/"

class return_sugarcrepe(Dataset):
    def __init__(self):
        all_pths = glob.glob(DATA_DIR)

        self.caps = []
        self.img_pths = []
        self.neg_caps = []

        for item in all_pths:
            # read json files
            f = open(item)
            json_obj = json.load(f)
            f.close()

            # iterate over each json file
            for _, subitem in enumerate(list(json_obj.keys())):
                self.caps.append(json_obj[subitem]['caption'])
                self.img_pths.append(os.path.join(IMG_DIR, "images", f"{json_obj[subitem]['filename']}.jpg"))
                self.neg_caps.append(json_obj[subitem]['negative_captions'])
    
    def __getitem__(self, index):
        # get image features
        img_pth = self.img_pths[index]
        img_out = Image.open(img_pth).convert('RGB')
        img_tensor = image_transform(img_out)

        # get text features
        txt = self.caps[index]
        txt_tensor = tokenizer(txt)

        neg_txt = self.neg_caps[index]
        neg_txt_tensor = tokenizer(neg_txt)

        return img_tensor, txt_tensor, neg_txt_tensor

    def __len__(self):
        return len(self.img_pths)






