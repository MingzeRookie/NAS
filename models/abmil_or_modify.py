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
    ):
        super(GatedAttention, self).__init__()
        self.L = input_size
        self.D = 512
        self.K = 2

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)
        self.z_u = nn.Linear(self.L * self.K, 1)
        self.z_sigma = nn.Linear(self.L * self.K, output_class-1)

        # ***********************************
        # *     Orginal Regression Head     *
        # ***********************************
        # b = []
        # for i in range(1,output_class):
            # b_i = torch.log(torch.tensor(1/(output_class/i-1)))
            # b.append(b_i)
        self.consecutive_value = 1
        b = self.consecutive_value*(np.arange(output_class-1) - np.arange(output_class-1).mean())
        # b_1 = torch.randn(torch.randn(),dtype=torch.float32)
        # a = torch.randn(output_class-2,dtype=torch.float32)
        # self.var_scaling = nn.Parameter(torch.tensor([b.max()+self.consecutive_value],dtype=torch.float32))
        self.var_scaling = nn.Parameter(torch.tensor([0],dtype=torch.float32))
        self.var_scaling.requires_grad_(False)
        # self.B = nn.Parameter(torch.concat([b_1, torch.cumsum(a**2,dim=0) + b_1], dim=0))
        self.B = nn.Parameter(torch.tensor(b,dtype=torch.float32))
        self.B.requires_grad_(False)
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
        z_u = self.z_u(M)
        # z_sigma = self.z_u(M)
        # prob, sigma = self.classifier(M)
        # z = (self.B - prob) * sigma * self.var_scaling
        # z = (self.B - z_u) * z_sigma**2 * self.var_scaling
        z = self.B - z_u
        z = torch.concat([self.bottom_boundary.unsqueeze(0),z,self.up_boundary.unsqueeze(0)],dim=1)
        cdf_pred_seq = torch.sigmoid(z)
        consecutive_prob = cdf_pred_seq[:,1:] - cdf_pred_seq[:,:-1]
        # print(consecutive_prob)
        return consecutive_prob, self.B, self.var_scaling
        # ordinal_seq = torch.cat(
        #     [self.bottom_boundary, self.cutpoints, self.up_boundary], dim=0
        # )
        # ordinal_seq = ordinal_seq.to(pred.device)
        # ordinal_seq = ordinal_seq.unsqueeze(0).repeat_interleave(repeats=B, dim=0)
        # ordinal_residual = torch.sigmoid(ordinal_seq[1:] - pred) - torch.sigmoid(
            # ordinal_seq[:-1] - pred
        # )
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        # return ordinal_residual, self.cutpoints, self.var_scaling
        


if __name__ == "__main__":
    model = GatedAttention(512, 4)
    x = torch.randn(1, 10, 512)
    print(model(x))
