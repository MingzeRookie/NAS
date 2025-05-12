# src/models/attention_layers.py
import torch
import torch.nn as nn

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None): # <--- 添加 key_padding_mask 参数
        # x shape: (batch_size, seq_len, embed_dim)
        # key_padding_mask shape: (batch_size, seq_len) - True 表示该位置应被忽略

        # MultiheadAttention expects query, key, value
        # 将传入的 mask 作为 key_padding_mask
        attn_output, _ = self.multihead_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_output) # Residual connection
        x = self.norm(x)
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads, dropout=0.1, embed_dim=None):
        super(CrossAttentionLayer, self).__init__()
        self.embed_dim = embed_dim if embed_dim is not None else query_dim
        self.num_heads = num_heads
        # self.dropout = dropout # dropout 参数在 MultiheadAttention 中使用

        self.q_proj = nn.Linear(query_dim, self.embed_dim)
        self.k_proj = nn.Linear(key_dim, self.embed_dim)
        self.v_proj = nn.Linear(key_dim, self.embed_dim)

        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(self.embed_dim) 
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, kv_padding_mask=None):
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        attn_output, _ = self.multihead_attn(q, k, v, key_padding_mask=kv_padding_mask)

        if query.shape == attn_output.shape:
            output = query + self.output_dropout(attn_output)
            output = self.norm(output)
        else: 
            output = self.norm(self.output_dropout(attn_output))
        return output