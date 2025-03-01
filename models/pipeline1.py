import torch
import torch.nn as nn
import numpy as np
from utils import CrossAttentionLayer, SelfAttentionLayer, FCNNBlock

class pipeline1(nn.Module):
    def __init__(self, embed_size, M, N, h, FEx, dropout):
        super(pipeline1, self).__init__()
        self.cross_attention = CrossAttentionLayer(embed_size, N, h, FEx, dropout)
        self.self_attention = SelfAttentionLayer(embed_size, M, h, FEx, dropout)
        self.fcnn = FCNNBlock(embed_size, FEx, embed_size, dropout)
    
    def forward(self, x, y, rx, ry):
        # get lower triangular mask
        trg_len = y.shape[-1]
        mask = torch.tril(torch.ones((trg_len, trg_len))).expand(1, trg_len, trg_len)
        mask = mask.to("cuda")

        # block 1
        x = self.cross_attention(self.self_attention(x, None), rx, mask)
        # block 2
        x = self.cross_attention(self.self_attention(x, None), ry, mask)
        # final
        out = self.fcnn(x)
        return out
    







