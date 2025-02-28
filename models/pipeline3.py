import torch
import torch.nn as nn
from utils import CrossAttentionLayer, SelfAttentionLayer, FCNNBlock

class pipeline3(nn.Module):
    def __init__(self, embed_size, M, N, h, FEx, dropout):
        super(pipeline3, self).__init__()
        self.self_attention = SelfAttentionLayer(embed_size, M, h, FEx, dropout)
        self.cross_attention = CrossAttentionLayer(embed_size, N, h, FEx, dropout)
        self.fcnn = FCNNBlock(embed_size, FEx, embed_size, dropout)
    
    def forward(self, x, y, rx, ry):
        # get lower triangular mask
        trg_len = y.shape[-1]
        mask = torch.tril(torch.ones((trg_len, trg_len))).expand(1, trg_len, trg_len)
        mask = mask.to("cuda")

        # block 1
        rt = self.cross_attention(rx, ry, ry, mask)
        x = self.cross_attention(self.self_attention(x, None), rx, mask)
        out = self.cross_attention(x, rt, rt, mask)

        # final
        out = self.fcnn(out)
        return out