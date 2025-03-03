import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(1, "/data/aditya/JANe/models/")
from utils import CrossAttentionLayer, SelfAttentionLayer, FCNNBlock

class pipeline1(nn.Module):
    def __init__(self, embed_size, M, N, h, FEx, dropout):
        super(pipeline1, self).__init__()
        self.cross_attention = CrossAttentionLayer(embed_size, N, h, FEx, dropout)
        self.self_attention = SelfAttentionLayer(embed_size, M, h, FEx, dropout)
        self.fcnn = FCNNBlock(embed_size, FEx, embed_size, dropout)
    
    def forward(self, x, y, rx, ry):
        print("Sizes", x.shape, rx.shape, ry.shape)
        x = x.reshape((x.size(0), 1, x.size(1)))

        # block 1
        x = self.cross_attention(self.self_attention(x, False), rx, True)
        # block 2
        x = self.cross_attention(self.self_attention(x, False), ry, True)
        # final
        out = self.fcnn(x)
        return out
    







