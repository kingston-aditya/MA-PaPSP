import torch
import torch.nn as nn
from utils import CrossAttentionLayer, FCNNBlock

class pipeline2(nn.Module):
    def __init__(self, embed_size, M, N, h, FEx, dropout):
        super(pipeline2, self).__init__()
        self.cross_attention = CrossAttentionLayer(embed_size, N, h, FEx, dropout)
        self.fcnn = FCNNBlock(embed_size, FEx, embed_size, dropout)
    
    def forward(self, x, y, rx, ry):
        # get lower triangular mask
        trg_len = y.shape[1]
        mask = torch.tril(torch.ones((trg_len, trg_len))).expand(1, trg_len, trg_len)
        mask = mask.to("cuda")

        # block 1
        x = self.cross_attention(x, rx, mask)
        # block 2
        x = self.cross_attention(x, ry, mask)
        # final
        out = self.fcnn(x)
        return out