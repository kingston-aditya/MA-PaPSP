import torch
from PIL import Image
import numpy as np
import os
from cc12_dataloader import return_cc12_train_dataset
from config import get_config
conf = get_config()
import sys
sys.path.insert(1, conf["repo_dir"])
from utils.run_clip import openclip_embeds
# import pdb 

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def main(config, batch):
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


if __name__ == "__main__":
    config = get_config()
    dataset_train = return_cc12_train_dataset()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=1, rank=0, shuffle=True
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config["batch_size"],
        num_workers=1,
        drop_last=True,
    )

    # import pdb 
    # pdb.set_trace()
    main(config, data_loader_train)





