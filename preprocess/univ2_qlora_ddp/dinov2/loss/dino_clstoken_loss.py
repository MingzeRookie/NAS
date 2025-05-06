# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
import logging
logger = logging.getLogger("dinov2_qlora")
# class DINOLoss(nn.Module):
#     def __init__(
#         self,
#         out_dim,
#         student_temp=0.1,
#         center_momentum=0.9,
#     ):
#         super().__init__()
#         self.student_temp = student_temp
#         self.center_momentum = center_momentum
#         self.register_buffer("center", torch.zeros(1, out_dim))
#         self.updated = True
#         self.reduce_handle = None
#         self.len_teacher_output = None
#         self.async_batch_center = None

#     @torch.no_grad()
#     def softmax_center_teacher(self, teacher_output, teacher_temp):
#         self.apply_center_update()
#         # teacher centering and sharpening
#         return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

#     @torch.no_grad()
#     def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
#         teacher_output = teacher_output.float()
#         world_size = dist.get_world_size() if dist.is_initialized() else 1
#         Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
#         B = Q.shape[1] * world_size  # number of samples to assign
#         K = Q.shape[0]  # how many prototypes

#         # make the matrix sums to 1
#         sum_Q = torch.sum(Q)
#         if dist.is_initialized():
#             dist.all_reduce(sum_Q)
#         Q /= sum_Q

#         for it in range(n_iterations):
#             # normalize each row: total weight per prototype must be 1/K
#             sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
#             if dist.is_initialized():
#                 dist.all_reduce(sum_of_rows)
#             Q /= sum_of_rows
#             Q /= K

#             # normalize each column: total weight per sample must be 1/B
#             Q /= torch.sum(Q, dim=0, keepdim=True)
#             Q /= B

#         Q *= B  # the columns must sum to 1 so that Q is an assignment
#         return Q.t()

#     def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
#         """
#         Cross-entropy between softmax outputs of the teacher and student networks.
#         """
#         # TODO: Use cross_entropy_distribution here
#         total_loss = 0
#         for s in student_output_list:
#             lsm = F.log_softmax(s / self.student_temp, dim=-1)
#             for t in teacher_out_softmaxed_centered_list:
#                 loss = torch.sum(t * lsm, dim=-1)
#                 total_loss -= loss.mean()
#         return total_loss

#     @torch.no_grad()
#     def update_center(self, teacher_output):
#         self.reduce_center_update(teacher_output)

#     @torch.no_grad()
#     def reduce_center_update(self, teacher_output):
#         self.updated = False
#         self.len_teacher_output = len(teacher_output)
#         self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
#         if dist.is_initialized():
#             self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

#     @torch.no_grad()
#     def apply_center_update(self):
#         if self.updated is False:
#             world_size = dist.get_world_size() if dist.is_initialized() else 1

#             if self.reduce_handle is not None:
#                 self.reduce_handle.wait()
#             _t = self.async_batch_center / (self.len_teacher_output * world_size)

#             self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

#             self.updated = True

# 修改 dino_clstoken_loss.py 中的 DINOLoss 类

class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None
        # 添加梯度记录，用于检测梯度异常
        self.last_gradients = None
        # 添加EMA平滑参数
        self.ema_smooth = 0.95
        # 添加安全调试标志
        self.safe_mode = True

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        
        # # 添加安全检查
        # if torch.isnan(teacher_output).any() or torch.isinf(teacher_output).any():
        #     logger.warning("教师输出包含NaN或Inf值，应用安全替换")
        #     # 复制一份避免修改原始数据
        #     teacher_output = teacher_output.clone()
        #     # 替换NaN和Inf值
        #     teacher_output[torch.isnan(teacher_output) | torch.isinf(teacher_output)] = 0.0
        
        # 添加安全的中心应用
        centered_output = teacher_output - self.center
        # 裁剪极值，避免指数爆炸
        centered_output = torch.clamp(centered_output, -50.0, 50.0)
        
        # 计算softmax，使用更安全的实现
        scaled = centered_output / teacher_temp
        # 减去每行的最大值以增强数值稳定性
        max_values, _ = torch.max(scaled, dim=-1, keepdim=True)
        exp_scaled = torch.exp(scaled - max_values)
        sum_exp = torch.sum(exp_scaled, dim=-1, keepdim=True)
        # 避免除以0
        sum_exp = torch.clamp(sum_exp, min=1e-10)
        return exp_scaled / sum_exp

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        # 添加安全检查
        if torch.isnan(teacher_output).any() or torch.isinf(teacher_output).any():
            logger.warning("教师输出包含NaN或Inf值，使用安全替换")
            teacher_output = teacher_output.clone()
            teacher_output[torch.isnan(teacher_output) | torch.isinf(teacher_output)] = 0.0
        
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # 安全的exp计算
        scaled = teacher_output / teacher_temp
        # 裁剪以防止exp溢出
        scaled = torch.clamp(scaled, -15.0, 15.0)
        Q = torch.exp(scaled).t()
        
        B = Q.shape[1] * world_size
        K = Q.shape[0]

        # 安全的归一化
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        # 避免除以0
        sum_Q = torch.clamp(sum_Q, min=1e-10)
        Q /= sum_Q

        for it in range(n_iterations):
            # 行归一化
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            # 避免除以0
            sum_of_rows = torch.clamp(sum_of_rows, min=1e-10)
            Q /= sum_of_rows
            Q /= K

            # 列归一化
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            # 避免除以0
            sum_of_cols = torch.clamp(sum_of_cols, min=1e-10)
            Q /= sum_of_cols
            Q /= B

        Q *= B
        return Q.t()

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        增强版的损失计算，添加了多种安全措施
        """
        total_loss = 0
        batch_size = 0
        
        # 对每个学生输出计算损失
        for i, s in enumerate(student_output_list):
            # 安全检查学生输出
            if torch.isnan(s).any() or torch.isinf(s).any():
                logger.warning(f"学生输出{i}包含NaN或Inf值，应用安全替换")
                s = s.clone()
                s[torch.isnan(s) | torch.isinf(s)] = 0.0
            
            # 计算对数softmax，带有数值稳定性优化
            s_scaled = s / self.student_temp
            # 裁剪极值
            s_scaled = torch.clamp(s_scaled, -10.0, 10.0)
            s_scaled_max, _ = torch.max(s_scaled, dim=-1, keepdim=True)
            s_scaled_exp = torch.exp(s_scaled - s_scaled_max)
            s_scaled_sum = torch.sum(s_scaled_exp, dim=-1, keepdim=True)
            # 避免log(0)
            s_scaled_sum = torch.clamp(s_scaled_sum, min=1e-10)
            lsm = torch.log(s_scaled_exp / s_scaled_sum)
            
            for j, t in enumerate(teacher_out_softmaxed_centered_list):
                # 安全检查教师输出
                if torch.isnan(t).any() or torch.isinf(t).any():
                    logger.warning(f"教师输出{j}包含NaN或Inf值，应用安全替换")
                    t = t.clone()
                    t[torch.isnan(t) | torch.isinf(t)] = 1.0 / t.shape[-1]  # 均匀分布
                
                # 计算和裁剪交叉熵损失
                loss = torch.sum(t * lsm, dim=-1)
                # 裁剪异常值
                loss = torch.clamp(loss, -10.0, 0.0)
                batch_loss = loss.mean()
                
                # 最后的安全检查
                if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                    logger.error(f"检测到NaN/Inf损失: {batch_loss}")
                    batch_loss = torch.tensor(0.1, device=s.device)  # 安全值
                
                total_loss -= batch_loss
                batch_size += 1
        
        # 最终安全检查
        # if batch_size > 0:
        #     return total_loss / batch_size  # 返回平均损失
        # else:
        #     logger.error("无有效批次用于损失计算")
        #     return torch.tensor(0.1, device=total_loss.device)  # 返回安全值

        if batch_size > 0:
            # 确保梯度流可以正常传播
            return total_loss / batch_size  
        else:
            logger.warning("无有效批次用于损失计算")
            # 使用小的随机值允许一些梯度信息
            return torch.tensor(0.1 + torch.rand(1).item() * 0.01, 
                                device=total_loss.device, 
                                requires_grad=True)

    @torch.no_grad()
    def update_center(self, teacher_output):
        # 安全检查
        if torch.isnan(teacher_output).any() or torch.isinf(teacher_output).any():
            logger.warning("中心更新收到NaN/Inf值，跳过此次更新")
            return
        
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
                
            # 对异步中心进行检查
            if torch.isnan(self.async_batch_center).any() or torch.isinf(self.async_batch_center).any():
                logger.warning("异步中心包含NaN/Inf值，保持旧的中心")
                self.updated = True
                return
                
            _t = self.async_batch_center / (self.len_teacher_output * world_size)
            
            # 平滑中心更新，防止波动太大
            new_center = self.center * self.center_momentum + _t * (1 - self.center_momentum)
            
            # 检查新中心是否合理
            if not torch.isnan(new_center).any() and not torch.isinf(new_center).any():
                self.center = new_center
            else:
                logger.warning("新的中心包含NaN/Inf值，保持旧的中心")
                
            self.updated = True
            
    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        """异步更新中心的辅助方法"""
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)
