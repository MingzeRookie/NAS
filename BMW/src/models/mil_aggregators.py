# src/models/mil_aggregators.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1024, dropout_rate=0.25): # 参数名与真实代码一致
        super(AttentionMIL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim # 这是 bottleneck 层的输出维度

        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1) # K=1

        # Bottleneck 层，用于在注意力加权求和后进行特征变换和降维/升维
        # 注意：这里的输入是 input_dim (因为 M 是 bag_feats 的加权平均)
        # 输出是 self.output_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, output_dim), # 输入维度是 input_dim
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, bag_feats, instance_mask=None): # <--- 添加 instance_mask 参数
        # bag_feats: (B, N, D) where B is batch_size, N is number of instances, D is feature dimension
        # instance_mask: (B, N) bool, True for valid instances, False for padding
        
        if bag_feats.dim() == 2: # 如果输入是 (N, D)，假定 batch_size = 1
            bag_feats = bag_feats.unsqueeze(0)

        # Calculate attention scores
        A_V = self.attention_V(bag_feats)  # (B, N, H)
        A_U = self.attention_U(bag_feats)  # (B, N, H)
        att_raw = self.attention_weights(A_V * A_U) # (B, N, 1) element-wise multiplication
        
        if instance_mask is not None:
            # 确保 instance_mask 的形状是 (B, N)
            if instance_mask.dim() == 1 and bag_feats.shape[0] == 1: # 如果 mask 是 (N) 且 batch_size 为 1
                instance_mask = instance_mask.unsqueeze(0) # 变为 (1, N)
            
            # instance_mask: True for valid. We want to set weights of PADDED (False) elements to -inf
            # unsqueeze(-1) to make it (B, N, 1) to broadcast with att_raw
            att_raw.masked_fill_(~instance_mask.bool().unsqueeze(-1), float('-inf'))
        
        # A_softmax: (B, N, 1)
        A_softmax = F.softmax(att_raw, dim=1)  # Softmax over instances in the bag (dim=1 for N)
        
        # Weighted sum of features
        # A_softmax transposed: (B, 1, N)
        # bag_feats: (B, N, D)
        # M: (B, 1, D)
        M = torch.bmm(A_softmax.transpose(1, 2), bag_feats) 
        M = M.squeeze(1)  # (B, D)
        
        # Pass through bottleneck layer
        final_feature = self.bottleneck(M) # (B, output_dim)
        
        return final_feature, A_softmax.squeeze(-1) # Return feature (B, output_dim) and attention scores (B, N)
