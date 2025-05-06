import torch
import torch.nn as nn


class OrdinalRegressionLoss(nn.Module):

    def __init__(
        self,
        num_class,
        var_scaling=2,
        cutp_scale=2,
        train_cutpoints=False,
        train_var_scaling=False,
    ):
        super().__init__()
        # scaling the input of preds, equals to scaling the sigmoid function
        self.var_scaling = torch.tensor(var_scaling, dtype=torch.float32)
        # initial cutpoints around zero and scaling
        self.cutpoints = torch.tensor(
            cutp_scale * (torch.arange(1, num_class) - num_class * 0.5),
            dtype=torch.float32,
        )

        if train_cutpoints:
            self.cutpoints = nn.Parameter(self.cutpoints)
            self.cutpoints.requires_grad_(True)
        if train_var_scaling:
            self.var_scaling = nn.Parameter(self.var_scaling)
            self.var_scaling.requires_grad_(True)
        self.up_boundary = torch.tensor([torch.inf], dtype=torch.float32)
        self.bottom_boundary = torch.tensor([-torch.inf], dtype=torch.float32)
        self.epsilon = 1e-15
        # self.epsilon = torch.tensor(torch.inf)
        self.nllloss = nn.NLLLoss()

    def forward(self, pred, label):

        B, _ = pred.shape
        pred = pred * self.var_scaling
        ordinal_seq = torch.cat(
            [self.bottom_boundary, self.cutpoints, self.up_boundary], dim=0
        )
        ordinal_seq = ordinal_seq.to(pred.device)
        ordinal_seq = ordinal_seq.unsqueeze(0).repeat_interleave(repeats=B, dim=0)
        ordinal_residual = torch.sigmoid(ordinal_seq[:, 1:] - pred) - torch.sigmoid(
            ordinal_seq[:, :-1] - pred
        )
        # print(ordinal_residual)
        # likelihoods = torch.clamp(ordinal_residual, self.epsilon, 1 - self.epsilon)
        log_likelihood = torch.log(ordinal_residual)
        # print(log_likelihood, label)
        # print(ordinal_residual, label)
        if label is None:
            loss = 0
        else:
            # loss = -torch.gather(log_likelihood, 1, label).mean()
            loss = self.nllloss(log_likelihood, label)

        return loss, ordinal_residual, self.cutpoints, self.var_scaling
