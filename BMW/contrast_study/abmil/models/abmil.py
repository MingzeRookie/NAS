# 文件: /remote-home/share/lisj/Workspace/SOTA_NAS/models/gated_attention_mil.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# 1. 这是您提供的 GatedAttention 模型，经过了批处理改造
class GatedAttention(nn.Module):
    def __init__(self, input_size, output_class):
        super(GatedAttention, self).__init__()
        self.L = input_size
        self.D = 512
        self.K = 2 # 注意力头的数量

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)
        
        # 分类器的输入维度是 K * L
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, output_class),
            # 注意：原代码的 nn.Sigmoid() 可能会导致数值不稳定，
            # 在 Lightning 中，我们通常在模型中输出原始 logits，
            # 然后在损失函数 nn.BCEWithLogitsLoss 中处理 sigmoid，这样更稳定。
            # 所以这里我把它移除了。
        )

    def forward(self, x):
        # x 的期望输入形状: (batch_size, num_patches, input_size)
        
        # 原代码中的 x.squeeze(0) 已移除，以支持批处理
        # H = x 
        
        A_V = self.attention_V(x)  # (B, N, D)
        A_U = self.attention_U(x)  # (B, N, D)
        A = self.attention_weights(A_V * A_U) # (B, N, K)
        A = torch.transpose(A, 2, 1)  # (B, K, N)
        A = F.softmax(A, dim=2)  # 在 num_patches 维度上进行 Softmax
        
        # torch.mm (矩阵乘法) -> torch.bmm (批处理矩阵乘法)
        M = torch.bmm(A, x)  # (B, K, L)
        
        # 调整 M 的形状以匹配分类器
        # B, K, L -> B, K*L
        M = M.view(x.size(0), -1)
        
        Y_prob = self.classifier(M)
        
        return Y_prob

# 2. 这是符合您项目框架的 PyTorch Lightning 封装器
class GatedAttentionMIL(pl.LightningModule):
    def __init__(self, input_size: int, output_class: int, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters() # 保存超参数，方便后续加载
        
        # 在 Lightning 模块内部实例化 GatedAttention 模型
        self.model = GatedAttention(input_size=input_size, output_class=output_class)
        
        # 定义损失函数
        # 使用 BCEWithLogitsLoss，因为它更数值稳定，且自带 sigmoid
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # 假设您的数据加载器返回一个包含特征和标签的字典或元组
        # 这里我们假设 batch['image'] 是特征，batch['label'] 是标签
        features = batch['image']
        labels = batch['label'].float() # 确保标签是 float 类型
        
        logits = self(features)
        loss = self.loss_fn(logits.squeeze(1), labels) # 使用 squeeze 以匹配维度
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features = batch['image']
        labels = batch['label'].float()
        
        logits = self(features)
        loss = self.loss_fn(logits.squeeze(1), labels)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer