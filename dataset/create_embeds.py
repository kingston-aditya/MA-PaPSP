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
import pdb 

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
    for i, curr_batch in enumerate(batch):
    # for i in range(len(batch["image"])):
        # get image embedding
        pdb.set_trace()
        # img_emd = clip_model.forward_img(curr_batch["image"][i])
        img_emd = clip_model.forward_img(curr_batch["image"])
        a[i] = img_emd.reshape(1,512)

        # get text embedding
        txt_emd = clip_model.forward_txt(curr_batch["caption"][i])
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
    dataset_train = return_cc12_train_dataset()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    # import pdb 
    # pdb.set_trace()
    main(config, batch)





