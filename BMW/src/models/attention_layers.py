# src/models/attention_layers.py
import torch
import torch.nn as nn

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        # MultiheadAttention expects query, key, value
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + self.dropout(attn_output) # Residual connection
        x = self.norm(x)
        return x

class CrossAttentionLayer(nn.Module): # 占位，未来实现
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # ...
    def forward(self, query, key, value, key_padding_mask=None):
        # ...
        pass