import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from torch.autograd import Variable

class GatedAttention(nn.Module):
    def __init__(self, input_size, output_class):
        super(GatedAttention, self).__init__()
        self.L = input_size
        self.D = 512
        self.K = 2

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, output_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        # print(A.shape)
        M = torch.mm(A, x)  # KxL
        M = M.view(-1,self.K*self.L)
        # print(M.shape)
        Y_prob = self.classifier(M)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob

if __name__ == "__main__":
    model = GatedAttention(512,4)
    x = torch.randn(1,10,512)
    print(model(x))
