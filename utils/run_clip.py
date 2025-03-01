import numpy as np
# from .dassl.coop import load_clip_to_cpu_name
# from .dassl.clip import clip
import torch
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor, CLIPTextModel
    
class clip_embeds(object):
    def __init__(self, backname="openai/clip-vit-base-patch16"):
        super(clip_embeds, self).__init__()
        self.device = "cuda"
        self.clip_txt_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.clip_img_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    def forward_img(self, image):
        tokens = self.processor(images=image, return_tensors="pt", padding=True)
        tokens.to(self.device)
        with torch.no_grad():
            output = self.clip_img_model(**tokens)
        image_features = output.last_hidden_state
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().detach().numpy()
        # size = 1 x (# of tokens) x embed size

    def forward_txt(self, txt):
        tokens = self.tokenizer(txt,return_tensors="pt", padding=True, truncation=True)
        tokens.to(self.device)
        with torch.no_grad():
            output = self.clip_txt_model(**tokens)
        text_features = output.last_hidden_state
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().detach().numpy()
        # size = 1 x (# of tokens) x embed size




