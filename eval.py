import torch
import numpy as np
import torch.nn as nn
from models import pipeline1, pipeline2, pipeline3
from configs.config import get_config
import os
import tqdm as tqdm
from torch.utils.data import DataLoader
from .dataset.dataset import COCO_CC12M
from sklearn.metrics import auc

config = get_config()

def score(img_feat, txt_feat):
    logits = img_feat @ txt_feat.T
    return logits

def sec_classify(scores, gt_scrs):
    ind_sc = np.argsort(scores)[::-1]
    srt_sc = scores[ind_sc]
    total_samples = srt_sc.shape[0]

    cov = []
    rsk = []
    for i in range(total_samples):
        if i%500==0:
            print("sec class",i)
        covrd = np.where(scores >= srt_sc[i])[0]
        cov.append(len(covrd)/total_samples)
        temp_scrs = gt_scrs[covrd]
        if len(covrd) > 0:
            rsk.append((temp_scrs[temp_scrs < 0.06].shape[0])/len(covrd))
        else:
            rsk.append(0)

    return cov, rsk, srt_sc

def eval(model, val_dataloader):
    checkpoint = torch.load(os.path.join(config["model_folder"],config["model_filename"]+str(5)), weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    device = "cuda"

    batch_iterator = tqdm(val_dataloader, desc=f"Processing epoch {epoch:02d}")
    sc = []
    for batch in batch_iterator:
        # get the data
        X = batch["input"].to(device)
        Y = batch["output"].to(device)
        RX = batch["ret_input"].to(device)
        RY = batch["ret_output"].to(device)

        # make prediction
        with torch.no_grad():
            out = model.forward(X, Y, RX, RY)
        sc.append(score(out,Y))
    return sc

if __name__ == "__main__":
    # get the dataloader
    val_dataloader = DataLoader(COCO_CC12M(config["retrieval_size"]), batch_size=1, shuffle=True)
    
    # get model
    if config["pipeline"] == 1:
        model = pipeline1.pipeline1(config["embed_size"], config["SA_number"], config["CA_number"])
    elif config["pipeline"] == 2:
        model = pipeline2.pipeline2(config["embed_size"], config["SA_number"], config["CA_number"])
    elif config["pipeline"] == 3:
        model = pipeline3.pipeline3(config["embed_size"], config["SA_number"], config["CA_number"])
    else:
        print("NOT a NUMBER")

    # for evaluation
    gt_scrs = # Get the CIDER scores
    sc = eval(model, val_dataloader)
    cov, rsk, _ = sec_classify(sc, gt_scrs)
    print("AUC", auc(cov, rsk))
    

    

