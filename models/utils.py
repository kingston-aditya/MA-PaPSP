import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim  # Dimension of input embeddings (d_model)
        self.num_heads = num_heads  # Number of attention heads
        self.head_dim = embed_dim // num_heads  # Dimension per head (d_k = d_v)

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear layers for query, key, value projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, y, causal_mask=False):
        # x: (batch_size, seq_length_x, embed_dim) - Query sequence
        # y: (batch_size, seq_length_y, embed_dim) - Key/Value sequence
        print(x.size())
        batch_size, seq_length_x, _ = x.size()
        _, seq_length_y, _ = y.size()
        
        # Project queries, keys, and values
        q = self.query_proj(x).reshape(batch_size, seq_length_x, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(y).reshape(batch_size, seq_length_y, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(y).reshape(batch_size, seq_length_y, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Apply causal mask if needed
        if causal_mask:
            mask = torch.tril(torch.ones(seq_length_x, seq_length_y, device=x.device)).unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_length_x, head_dim)
        
        # Concatenate heads and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length_x, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(CrossAttentionBlock, self).__init__()
        self.attention = CrossAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ffn = nn.Sequential(
            nn.Linear(embed_size, int(forward_expansion*embed_size)),
            nn.ReLU(),
            nn.Linear(int(forward_expansion*embed_size), embed_size)
            )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, mask):
        query.to("cuda")
        key.to("cuda")
        attention = self.attention(query, key, mask)
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
            x = layer(x, x, src_mask)
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
            x = layer(xq, xv, src_mask)
        return x
    


        

        




        

        
