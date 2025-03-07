from transformers import AutoProcessor, AutoModel
import torch

class siglip_embeds(object):
    def __init__(self, model_name, pretrain, device):
        self.device = device
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    
    def preprocessor(self, txt=None, img=None):
        if img is None:
            inputs = self.processor(text=[txt], padding="max_length", return_tensors="pt")
        else:
            inputs = self.processor(text=img, padding="max_length", return_tensors="pt")
        return inputs
    
    def forward_img(self, img_tensor):
        img_tensor.to(self.device)
        with torch.no_grad():
            image_features = self.model(**img_tensor)
        return image_features.cpu().detach()

    def forward_txt(self, txt_tensor):
        txt_tensor.to(self.device)
        with torch.no_grad():
            text_features = self.model(**txt_tensor)
        return text_features.cpu().detach()