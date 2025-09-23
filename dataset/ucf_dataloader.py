from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as v2
from tokenizer import SimpleTokenizer

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

DATA_DIR = "/nfshomes/asarkar6/trinity/JANe-project/ucf101/"

def prompt_creater(cat):
    name = "An image of {}, an action.".format(cat)
    return name

# ucf101 test set 
class return_ucf(Dataset):
    def __init__(self):
        # get the val and test split
        f1 = open(os.path.join(DATA_DIR, "split_zhou_UCF101.json"))
        self.json_obj = json.load(f1)
        f1.close()
    
    def __getitem__(self, index):
        # get paths
        img_pth = os.path.join(DATA_DIR, "UCF-101-midframes", self.json_obj["test"][index][0])

        # get image features
        img_out = Image.open(img_pth).convert('RGB')
        img_tensor = image_transform(img_out)

        # get text features
        txt = prompt_creater(self.json_obj["test"][index][-1])
        txt_tensor = tokenizer(txt)

        return img_tensor, txt_tensor

    def __len__(self):
        return len(self.json_obj["test"])
    
# ucf101 train set
class return_ucf_train(Dataset):
    def __init__(self):
        # get the val and test split
        f1 = open(os.path.join(DATA_DIR, "split_zhou_UCF101.json"))
        self.json_obj = json.load(f1)
        f1.close()
    
    def __getitem__(self, index):
        # get paths
        img_pth = os.path.join(DATA_DIR, "UCF-101-midframes", self.json_obj["train"][index][0])

        # get image features
        img_out = Image.open(img_pth).convert('RGB')
        img_tensor = image_transform(img_out)

        # get text features
        txt = prompt_creater(self.json_obj["train"][index][-1])
        txt_tensor = tokenizer(txt)

        return img_tensor, txt_tensor

    def __len__(self):
        return len(self.json_obj["train"])






