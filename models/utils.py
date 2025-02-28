import torch
import torch.nn as nn
import math

class retrieve_XY(object):
    def __init__(self,Xr,Yr,k):
        super(retrieve_XY,self).__init__()
        self.device = torch.device('cuda')
        self.Xr = torch.from_numpy(Xr).float().to(self.device)
        self.Yr = torch.from_numpy(Yr).float().to(self.device)
        self.k = k

    def retrieve_X(self,q):
        q = torch.from_numpy(q).float().to(self.device)
        ans = torch.matmul(self.Xr,q)
        _,ind = torch.topk(ans,self.k)
        ind = ind.cpu().detach().numpy()
        return ind

    def retrieve_Y(self,q):
        q = torch.from_numpy(q).float().to(self.device)
        ans = torch.matmul(self.Yr,q)
        _,ind = torch.topk(ans,self.k)
        ind = ind.cpu().detach().numpy()
        return ind  

class CrossAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size//heads

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)

        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        head_dim = query.shape[-1]
        attention_scores = (query @ key.transpose(-1, -1))/ math.sqrt(head_dim)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores

    def forward(self, query, keys, values, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # reshape everything
        values = values.reshape(N, value_len, self.embed_size)
        keys = keys.reshape(N, key_len, self.embed_size)
        queries = query.reshape(N, query_len, self.embed_size)

        # add the heads of transformer
        query_fin = self.queries(queries)
        query_fin = query_fin.view(query_fin.shape[0], query_fin.shape[1], self.heads, self.head_dim).transpose(1,2)
        
        keys_fin = self.keys(keys)
        keys_fin = keys_fin.view(keys_fin.shape[0], keys_fin.shape[1], self.heads, self.head_dim).transpose(1,2)

        values_fin = self.values(values)
        values_fin = values_fin.view(values_fin.shape[0], values_fin.shape[1], self.heads, self.head_dim).transpose(1,2)

        # get the attention scores
        x, self.attention_scores = CrossAttention.attention(query_fin, keys_fin, values_fin, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.embed_size)

        out = self.fc_out(x)
        return out  

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(CrossAttentionBlock, self).__init__()
        self.attention = CrossAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ffn = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
            )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        attention = self.attention(query, key, value, mask)
        x = self.dropout(self.norm1(attention + query))
        xf = self.ffn(x)
        out = self.dropout(self.norm2(xf + x))
        return out

class FCNNBlock(nn.Module):
    def __init__(self, embed_size, forward_expansion, final_size, dropout):
        super(FCNNBlock, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, int(forward_expansion*embed_size)),
            nn.ReLU(),
            nn.Linear(int(forward_expansion*embed_size), final_size)
            )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(final_size)
    
    def forward(self, x):
        xf = self.ffn(x)
        out = self.dropout(self.norm1(xf+x))
        return out
    

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout):
        super(SelfAttentionLayer, self).__init__()
        self.device = "cuda"
        self.heads = heads
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(embed_size, heads, forward_expansion, dropout)
             for _ in range(num_layers)]
        )
    
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, x, x, src_mask)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout):
        super(CrossAttentionLayer, self).__init__()
        self.device = "cuda"
        self.heads = heads
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(embed_size, heads, forward_expansion, dropout)
            for _ in range(num_layers)]
        )        

    def forward(self, xq, xv, src_mask):
        for layer in self.layers:
            x = layer(xq, xv, xv, src_mask)
        return x
    


        

        




        

        
