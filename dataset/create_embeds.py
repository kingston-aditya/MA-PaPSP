import torch
from PIL import Image
import numpy as np
import os

# from cc12_cc3_sbu_dataloader import return_cc12_cc3_sbu
from pets_dataloader import return_pets, return_pets_train

import sys
sys.path.insert(1, conf["repo_dir"])
from utils.run_clip import openclip_embeds
import timm
# import pdb 

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def get_cc12m(config, batch):
    a = {}; b= {}
    clip_model = openclip_embeds(config["model_type"], config["pretrain"], config["device"])
    for i, curr_batch in enumerate(batch):
        batch_size = config["batch_size"]

        # get image embedding
        img_emd = clip_model.forward_img(curr_batch["image"].to("cuda"))
        a[i] = img_emd.reshape(batch_size,512)

        # get text embedding
        txt_emd = clip_model.forward_txt(curr_batch["caption"].to("cuda"))
        b[i] = txt_emd.reshape(batch_size,512)

        # save files based on numbers
        if i%conf["shard_num"] == 0:
            print("done", int(i//conf["shard_num"]))
            out_fil_img = os.path.join(config["out_pth"],"cc12m_img_shard"+str(int(i//conf["shard_num"])))
            out_fil_txt = os.path.join(config["out_pth"],"cc12m_txt_shard"+str(int(i//conf["shard_num"])))
            np.save(out_fil_img, np.asarray(list(a.values())))
            np.save(out_fil_txt, np.asarray(list(b.values())))
            a={}; b={}

def get_cc12m_cc3m_sbu(config, batch):
    batch_size = config["batch_size"]
    b = {}
    clip_model = openclip_embeds(config["model_type"], config["pretrain"], config["device"])
    for i, curr_batch in enumerate(batch):

        # get text embedding
        txt_emd = clip_model.forward_txt(curr_batch.to("cuda"))
        b[i] = torch.reshape(txt_emd, (batch_size,512))

        # save files based on numbers
        print("done", i)
        out_fil_txt = os.path.join(config["out_pth"],"cc12m_txt_shard_"+str(config["pretrain"])+str(int(i))+".pt")
        torch.save(b[i], out_fil_txt)

def get_classify(config, batch):
    a = {}; b= {}
    clip_model = openclip_embeds(config["model_type"], config["pretrain"], config["device"])
    for i, curr_batch in enumerate(batch):
        batch_size = config["batch_size"]

        # get outputs
        img_toks, txt_toks = curr_batch

        # get image embedding
        img_emd = clip_model.forward_img(img_toks.to("cuda"))
        a[i] = img_emd.reshape(batch_size,512)

        # get text embedding
        txt_emd = clip_model.forward_txt(txt_toks.to("cuda"))
        b[i] = txt_emd.reshape(batch_size,512)

        print("done", i)
        # print("path", "flowers_img_shard_"+str(config["pretrain"])+str(int(i))+".pt")
        out_fil_img = os.path.join(config["out_pth"],"pets_img_shard_"+str(config["pretrain"])+str(int(i))+".pt")
        out_fil_txt = os.path.join(config["out_pth"],"pets_txt_shard_"+str(config["pretrain"])+str(int(i))+".pt")
        torch.save(a[i], out_fil_img)
        torch.save(b[i], out_fil_txt)


if __name__ == "__main__":
    config = get_config()
    dataset_train = return_pets_train()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=1, rank=0, shuffle=True
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config["batch_size"],
        num_workers=12,
        drop_last=True,
    )

    # import pdb 
    # pdb.set_trace()
    get_classify(config, data_loader_train)





