import torch
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import os
from config import get_config
from cc12_dataloader import return_cc12_train_dataset

conf = get_config()
import sys
sys.path.insert(1, conf["repo_dir"])
from utils.run_clip import clip_embeds


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def trans_form(n_px=224):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def load_img(pth, form_ten):
    img = Image.open(pth)
    img_enc = form_ten(img).to("cuda")
    img_res = torch.reshape(img_enc, (1,3,224,224))
    return img_res

def main(config, batch):
    a = {}; b= {}
    clip_model = clip_embeds()
    for i in range(len(batch["image"])):
        # get image embedding
        img_emd = clip_model.forward_img(batch["image"][i])
        a[i] = img_emd.reshape(1,512)

        # get text embedding
        txt_emd = clip_model.forward_txt(batch["caption"][i])
        b[i] = txt_emd.reshape(1,512)

        # save files based on numbers
        if i%10e5 == 0:
            print("done", int(i//10e5))
            out_fil_img = os.path.join(config["out_pth"],"cc12m_img_shard"+str(int(i//10e5)))
            out_fil_txt = os.path.join(config["out_pth"],"cc12m_txt_shard"+str(int(i//10e5)))
            np.save(out_fil_img, np.asarray(list(a.values())))
            np.save(out_fil_txt, np.asarray(list(b.values())))

if __name__ == "__main__":
    # list of file paths of img and txt
    # fil_img_lst = [os.path.join(d["fil_pth"],i) for i in sorted(os.listdir(d["fil_pth"])) if i[-4:]=='.jpg']
    # fil_txt_lst = [os.path.join(d["fil_pth"],i) for i in sorted(os.listdir(d["fil_pth"])) if i[-4:]=='.txt']
    config = get_config()
    batch = return_cc12_train_dataset()
    main(config, batch)





