import numpy as np
from .dassl.coop import load_clip_to_cpu_name
from .dassl.clip import clip
import torch

class clip_embeds(object):
    def __init__(self, backname="ViT-B/16"):
        super(clip_embeds, self).__init__()
        self.device = "cuda"
        clip_model = load_clip_to_cpu_name(backname)
        clip_model.to(self.device)
        self.clip_model = clip_model
    
    def forward_img(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().detach().numpy()
    
    def forward_txt(self, txt):
        txt_tokens = clip.tokenize(txt)
        txt_features = self.clip_model.encode_text(torch.Tensor(txt_tokens).to(self.device))
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        return txt_features.cpu().detach().numpy()

