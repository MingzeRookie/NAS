import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from torch.autograd import Variable


class GatedAttention(nn.Module):
    def __init__(
        self,
        input_size,
        output_class,
        var_scaling=2,
        cutp_scale=2,
        train_cutpoints=True,
        train_var_scaling=True,
    ):
        super(GatedAttention, self).__init__()
        self.L = input_size
        self.D = 512
        self.K = 2

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)
        self.classifier = nn.Linear(self.L * self.K, 1)

        # ***********************************
        # *     Orginal Regression Head     *
        # ***********************************
        # scaling the input of preds, equals to scaling the sigmoid function
        self.var_scaling = torch.tensor(var_scaling, dtype=torch.float32)
        # initial cutpoints around zero and scaling
        self.cutpoints = torch.tensor(
            cutp_scale * (torch.arange(1, output_class) - output_class * 0.5),
            dtype=torch.float32,
        )
        if train_cutpoints:
            self.cutpoints = nn.Parameter(self.cutpoints)
            self.cutpoints.requires_grad_(True)
        if train_var_scaling:
            self.var_scaling = nn.Parameter(self.var_scaling)
            self.var_scaling.requires_grad_(True)
        self.up_boundary = nn.Parameter(torch.tensor([torch.inf], dtype=torch.float32))
        self.up_boundary.requires_grad_(False)
        self.bottom_boundary = nn.Parameter(torch.tensor([-torch.inf], dtype=torch.float32))
        self.bottom_boundary.requires_grad_(False)

    def forward(self, x):

        x = x.squeeze(0)
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        # print(A.shape)
        M = torch.mm(A, x)  # KxL
        M = M.view(-1, self.K * self.L)
        # print(M.shape)
        Y_prob = self.classifier(M)
        pred = Y_prob * self.var_scaling
        ordinal_seq = torch.cat(
            [self.bottom_boundary, self.cutpoints, self.up_boundary], dim=0
        )
        ordinal_seq = ordinal_seq.to(pred.device)
        # ordinal_seq = ordinal_seq.unsqueeze(0).repeat_interleave(repeats=B, dim=0)
        ordinal_residual = torch.sigmoid(ordinal_seq[1:] - pred) - torch.sigmoid(
            ordinal_seq[:-1] - pred
        )
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        return ordinal_residual, self.cutpoints, self.var_scaling


if __name__ == "__main__":
    model = GatedAttention(512, 4)
    x = torch.randn(1, 10, 512)
    print(model(x))
