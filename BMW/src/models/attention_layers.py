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

class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads, dropout=0.1, embed_dim=None): # embed_dim 通常等于 query_dim
        super(CrossAttentionLayer, self).__init__()
        self.embed_dim = embed_dim if embed_dim is not None else query_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(query_dim, self.embed_dim)
        self.k_proj = nn.Linear(key_dim, self.embed_dim)
        self.v_proj = nn.Linear(key_dim, self.embed_dim) # 通常 key 和 value 来自同一源，所以 key_dim

        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(self.embed_dim) # 或者 query_dim，取决于您是否希望输出维度与输入query一致
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, kv_padding_mask=None):
        # query: (batch_size, seq_len_q, query_dim)
        # key_value: (batch_size, seq_len_kv, key_dim)
        # kv_padding_mask: (batch_size, seq_len_kv) - True 表示该位置是 padding，应被忽略

        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        # attn_output: (batch_size, seq_len_q, embed_dim)
        attn_output, _ = self.multihead_attn(q, k, v, key_padding_mask=kv_padding_mask)

        # 残差连接和 LayerNorm (通常应用于 query)
        # 如果 query 的原始维度与 embed_dim 不同，您可能需要调整或添加额外的投影层
        # 这里假设 query_dim == embed_dim 或在 self.q_proj 后维度一致
        if query.shape == attn_output.shape: # 确保可以进行残差连接
            output = query + self.output_dropout(attn_output)
            output = self.norm(output)
        else: # 如果维度不匹配，可能只返回 attn_output 或进行投影
            output = self.norm(self.output_dropout(attn_output))

        return output