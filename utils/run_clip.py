import numpy as np
# from .dassl.coop import load_clip_to_cpu_name
# from .dassl.clip import clip
import open_clip
import torch
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor, CLIPTextModel

class openclip_embeds(object):
    def __init__(self, model_name, pretrain, device):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrain)
        self.model.to(self.device)
        self.model.eval()

    def preprocessor(self, img):
        return self.preprocess(img).unsqueeze().to("cuda")
    
    def forward_img(self, img_tensor):
        img_tensor.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
        return image_features.cpu().detach().numpy()

    def forward_txt(self, txt_tensor):
        txt_tensor.to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(txt_tensor)
        return text_features.cpu().detach().numpy()

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
        # tokens = self.tokenizer(txt,return_tensors="pt", padding=True, truncation=True)
        tokens = txt
        tokens.to(self.device)
        with torch.no_grad():
            output = self.clip_txt_model(**tokens)
        text_features = output.last_hidden_state
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().detach().numpy()
        # size = 1 x (# of tokens) x embed size




