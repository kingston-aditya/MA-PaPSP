from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as v2
from tokenizer import SimpleTokenizer
from config import get_config
config = get_config()
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

def prompt_creater(cat):
    name = "An image of {}, a flower.".format(cat)
    return name

class return_flowers(Dataset):
    def __init__(self):
        # get the val and test split
        f1 = open(os.path.join(config["data_dir"], "split_zhou_OxfordFlowers.json"))
        self.json_obj = json.load(f1)
        f1.close()

        # get cats to labels
        f2 = open(os.path.join(config["data_dir"], "cat_to_name.json"))
        self.cat_obj = json.load(f2)
        f2.close()
    
    def __getitem__(self, index):
        # get paths
        img_pth = os.path.join("/data/datasets/oxford_flowers/jpg/", self.json_obj["test"][index][0])

        # get image features
        img_out = Image.open(img_pth).convert('RGB')
        img_tensor = image_transform(img_out)

        # get text features
        txt = prompt_creater(self.json_obj["test"][index][-1])
        txt_tensor = tokenizer(txt)

        return img_tensor, txt_tensor

    def __len__(self):
        return len(self.json_obj["test"])






