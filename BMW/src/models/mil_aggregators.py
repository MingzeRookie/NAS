# src/models/mil_aggregators.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1024, dropout_rate=0.25):
        super(AttentionMIL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, bag_feats):
        # bag_feats: (N, D) where N is number of instances in the bag, D is feature dimension
        # Or (1, N, D) if batch_size is 1
        if bag_feats.dim() == 2:
            bag_feats = bag_feats.unsqueeze(0) # Add batch dimension: (1, N, D)

        # Calculate attention scores
        A_V = self.attention_V(bag_feats)  # (1, N, H)
        A_U = self.attention_U(bag_feats)  # (1, N, H)
        A = self.attention_weights(A_V * A_U) # (1, N, 1) element-wise multiplication
        A = torch.transpose(A, 2, 1)  # (1, 1, N)
        A = F.softmax(A, dim=2)  # Softmax over instances in the bag

        # Weighted sum of features
        M = torch.bmm(A, bag_feats).squeeze(1)  # (1, 1, N) bmm (1, N, D) -> (1, 1, D) -> (1,D)
        
        # Pass through bottleneck layer
        final_feature = self.bottleneck(M) # (1, output_dim)
        return final_feature, A.squeeze(0) # Return feature and attention weights
