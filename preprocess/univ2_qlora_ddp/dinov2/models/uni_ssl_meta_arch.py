# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 您的组织名称
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import os

import torch
from torch import nn
import timm
import timm.layers
from torch.cuda.amp import GradScaler
import torch.distributed as dist

from dinov2.loss import DINOLoss, KoLeoLoss
from dinov2.layers import DINOHead
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups

from models.lora_adapter import apply_qlora, get_trainable_params

logger = logging.getLogger("dinov2_qlora")

class UNISSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 使用普通GradScaler代替ShardedGradScaler
        self.fp16_scaler = GradScaler() if cfg.compute_precision.grad_scaler else None
        # 添加训练步数计数器，用于center定期重置
        self.training_steps = 0
        self.center_reset_interval = 200  # 每200次迭代重置一次中心
        self.training_steps = 0

        # 保存原始模型权重以便初始化教师模型
        self.original_weights = None

        student_model_dict = dict()
        teacher_model_dict = dict()

        # 创建UNI-V2 encoder实例
        student_backbone, teacher_backbone, embed_dim = self._build_uni_v2_models(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        
        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()
        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        # 保存原始权重(在应用Q-LoRA前)用于初始化教师模型
        self.original_state_dict = {}
        for key, model in student_model_dict.items():
            self.original_state_dict[key] = model.state_dict()

        # 应用Q-LoRA到学生模型
        if hasattr(cfg.lora, 'enabled') and cfg.lora.enabled:
            logger.info(f"OPTIONS -- LoRA -- applying Q-LoRA with rank {cfg.lora.r}")
            for key, model in student_model_dict.items():
                converted = apply_qlora(
                    model,
                    r=cfg.lora.r,
                    alpha=cfg.lora.alpha,
                    dropout=cfg.lora.dropout,
                    target_modules=cfg.lora.target_modules,
                    bias=cfg.lora.bias,
                    bits=cfg.quantization.bits if hasattr(cfg.quantization, 'enabled') and cfg.quantization.enabled else 32,
                    group_size=cfg.quantization.group_size if hasattr(cfg.quantization, 'group_size') else 128,
                    double_quant=cfg.quantization.double_quant if hasattr(cfg.quantization, 'double_quant') else False,
                )
                logger.info(f"Converted {converted} layers of {key} to Q-LoRA")
            
            # 输出可训练参数信息
            for key, model in student_model_dict.items():
                trainable, total, percentage = get_trainable_params(model)
                logger.info(f"{key} - Trainable params: {trainable:,d} ({percentage}%) of {total:,d} total params")

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # 教师模型不需要梯度
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

        self._init_stability_measures()
                    
    def _build_uni_v2_models(self, cfg):
        """创建UNI-V2 encoder模型"""
        # 配置timm参数
        timm_kwargs = {
            'model_name': cfg.student.arch,
            'img_size': cfg.crops.global_crops_size,
            'patch_size': cfg.student.patch_size,
            'depth': cfg.student.depth,
            'num_heads': cfg.student.num_heads,
            'init_values': cfg.student.layerscale,
            'embed_dim': cfg.student.embed_dim,
            'mlp_ratio': cfg.student.mlp_ratio,
            'num_classes': 0,
            'no_embed_class': True,
            'global_pool': '',  # 关闭全局池化，使模型返回完整的token序列
            'mlp_layer': timm.layers.SwiGLUPacked if cfg.student.ffn_layer == "swiglu" else nn.GELU,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': cfg.student.num_register_tokens,
            'dynamic_img_size': True
        }
        
        logger.info("Creating UNI-V2 encoder student instance...")
        student = timm.create_model(**timm_kwargs)
        
        logger.info("Creating UNI-V2 encoder teacher instance...")
        teacher = timm.create_model(**timm_kwargs)
        
        # 加载预训练权重（如果提供）
        if cfg.student.pretrained_weights:
            logger.info(f"Loading pretrained weights from {cfg.student.pretrained_weights}")
            state_dict = torch.load(cfg.student.pretrained_weights, map_location="cpu")
            student.load_state_dict(state_dict, strict=True)
            teacher.load_state_dict(state_dict, strict=True)
            
        # 获取嵌入维度
        embed_dim = cfg.student.embed_dim
        
        return student, teacher, embed_dim
    
    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp, grad_accumulation_steps=4):
        """
        执行前向和反向传播，计算损失
        
        Args:
            images: 包含全局和局部裁剪的字典
            teacher_temp: 教师模型温度参数
            grad_accumulation_steps: 梯度累积步数
        
        Returns:
            损失字典和累积的损失值
        """
        # 获取数据
        n_global_crops = 2  # 全局裁剪数量固定为2
        n_local_crops = self.cfg.crops.local_crops_number  # 从配置获取局部裁剪数量
        
        # 将数据移至GPU
        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)
        
        # 计算损失项数量
        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops
        
        # 同步所有进程中的教师模型状态
        if dist.is_initialized():
            for k, v in self.teacher.items():
                for param in v.parameters():
                    dist.broadcast(param.data, 0)
        
        # 定义用于前向传播的安全函数
        def safe_forward(module, inputs):
            try:
                with torch.cuda.amp.autocast(enabled=self.fp16_scaler is not None):
                    return module(inputs)
            except RuntimeError as e:
                if "nan" in str(e).lower() or "inf" in str(e).lower():
                    logger.error(f"检测到NaN/Inf错误: {str(e)}")
                    # 返回零tensor作为安全值
                    if isinstance(inputs, list):
                        return torch.zeros_like(inputs[0])
                    return torch.zeros_like(inputs)
                else:
                    raise e
        
        # 检查tensor是否包含NaN或Inf
        def check_tensor(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logger.warning(f"{name} 包含NaN或Inf值，将被替换为安全值")
                # 替换为安全值
                tensor_safe = tensor.clone()
                tensor_safe[tensor_safe.isnan() | tensor_safe.isinf()] = 0.0
                return tensor_safe
            return tensor
        
        # 获取教师模型输出（无梯度）
        @torch.no_grad()
        def get_teacher_output():
            # 处理全局裁剪
            x, n_global_crops_teacher = global_crops, n_global_crops
            
            # 分批处理全局裁剪
            teacher_outputs = []
            
            for i in range(0, x.shape[0], n_global_crops_teacher):
                batch = x[i:i+n_global_crops_teacher]
                # 前向传播
                teacher_output = self.teacher.backbone(batch)
                
                # 获取CLS token
                cls_token = teacher_output[:, 0]
                teacher_outputs.append(cls_token)
            
            # 合并输出
            teacher_cls_tokens = torch.cat(teacher_outputs)
            
            # 通过教师头部
            teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
            
            # 处理维度
            if teacher_cls_tokens_after_head.dim() == 1:
                teacher_cls_tokens_after_head = teacher_cls_tokens_after_head.unsqueeze(0)
            
            # 定期重置中心点
            if self.training_steps % self.center_reset_interval == 0:
                # 计算新的中心
                new_center = torch.mean(teacher_cls_tokens_after_head, dim=0, keepdim=True)
                
                # 在分布式环境中同步中心值
                if dist.is_initialized():
                    dist.all_reduce(new_center)
                    new_center = new_center / dist.get_world_size()
                
                # 检查中心是否包含异常值
                if not torch.isnan(new_center).any() and not torch.isinf(new_center).any():
                    # 平滑更新中心
                    alpha = 0.95
                    old_center = self.dino_loss.center
                    self.dino_loss.center = alpha * old_center + (1 - alpha) * new_center
                    logger.info(f"重置DINO中心点，迭代次数: {self.training_steps}")
                else:
                    logger.warning(f"DINO中心点包含异常值，跳过更新")
            
            self.training_steps += 1
            
            # 执行centering操作
            if self.cfg.train.centering == "centering":
                # 安全性检查
                if teacher_cls_tokens_after_head.shape[0] == 0:
                    logger.error("教师输出为空，无法计算softmax")
                    default_shape = (n_global_crops_teacher, self.dino_out_dim)
                    return torch.ones(default_shape).to(teacher_cls_tokens_after_head.device) / self.dino_out_dim
                
                # 异常值检查
                if torch.isnan(teacher_cls_tokens_after_head).any() or torch.isinf(teacher_cls_tokens_after_head).any():
                    logger.error("教师输出包含NaN或Inf，使用安全默认值")
                    default_shape = (teacher_cls_tokens_after_head.shape[0], self.dino_out_dim)
                    return torch.ones(default_shape).to(teacher_cls_tokens_after_head.device) / self.dino_out_dim
                
                # 计算教师softmax输出
                teacher_dino_softmaxed_centered = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                )
                # 更新中心
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                )
            else:
                raise NotImplementedError("不支持的centering方法")
            
            # 将结果按global_crops分组
            teacher_outputs_list = []
            batch_size = teacher_dino_softmaxed_centered.shape[0] // n_global_crops_teacher
            
            for i in range(n_global_crops_teacher):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                teacher_outputs_list.append(teacher_dino_softmaxed_centered[start_idx:end_idx])
            
            return teacher_outputs_list
        
        # 获取教师模型输出
        teacher_dino_softmaxed_centered_list = get_teacher_output()
        
        # 初始化损失字典和累积器
        loss_dict = {}
        loss_accumulator = 0
        
        # === 处理学生模型输出 ===
        # 使用更可靠的批处理方式
        max_batch_size = 8  # 可调整的批大小
        
        # 处理全局裁剪
        global_batches = []
        for i in range(0, global_crops.shape[0], max_batch_size):
            batch = global_crops[i:i+max_batch_size]
            try:
                # 获取backbone
                backbone = self.student.backbone
                # 前向传播
                output = backbone(batch)
                # 提取CLS token
                global_batches.append(output[:, 0])
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("处理全局裁剪时遇到OOM，降低批次大小")
                    torch.cuda.empty_cache()
                    # 减小批次大小重试
                    for j in range(0, batch.shape[0], 2):
                        small_batch = batch[j:j+2]
                        output = backbone(small_batch)
                        global_batches.append(output[:, 0])
                else:
                    raise e
        
        # 合并全局裁剪输出
        student_global_cls_tokens = torch.cat(global_batches)
        
        # 处理局部裁剪
        local_batches = []
        for i in range(0, local_crops.shape[0], max_batch_size):
            batch = local_crops[i:i+max_batch_size]
            try:
                # 前向传播
                output = backbone(batch)
                # 提取CLS token
                local_batches.append(output[:, 0])
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("处理局部裁剪时遇到OOM，降低批次大小")
                    torch.cuda.empty_cache()
                    # 减小批次大小重试
                    for j in range(0, batch.shape[0], 2):
                        small_batch = batch[j:j+2]
                        output = backbone(small_batch)
                        local_batches.append(output[:, 0])
                else:
                    raise e
        
        # 合并局部裁剪输出
        student_local_cls_tokens = torch.cat(local_batches) if local_batches else torch.tensor([]).to(student_global_cls_tokens.device)
        
        # 通过DINO头部
        student_global_cls_tokens_after_head = self.student.dino_head(student_global_cls_tokens)
        student_local_cls_tokens_after_head = self.student.dino_head(student_local_cls_tokens) if student_local_cls_tokens.shape[0] > 0 else None
        
        # === 计算损失 ===
        # 计算局部裁剪损失
        if n_local_crops > 0 and student_local_cls_tokens_after_head is not None:
            # 确保局部裁剪数量是全局裁剪数量的整数倍
            n_local_crops_per_global = n_local_crops // n_global_crops
            if n_local_crops % n_global_crops != 0:
                logger.warning(f"局部裁剪数量({n_local_crops})不能被全局裁剪数量({n_global_crops})整除，将调整")
                n_local_crops_per_global = max(1, n_local_crops // n_global_crops)
            
            # 将局部裁剪分组
            student_local_chunks = []
            for i in range(n_global_crops):
                start_idx = i * n_local_crops_per_global
                end_idx = min(start_idx + n_local_crops_per_global, student_local_cls_tokens_after_head.shape[0])
                if end_idx > start_idx:
                    student_local_chunks.append(student_local_cls_tokens_after_head[start_idx:end_idx])
            
            # 确保教师输出和学生输出维度匹配
            teacher_chunks = []
            if isinstance(teacher_dino_softmaxed_centered_list, list):
                # 如果是列表，直接使用
                teacher_chunks = teacher_dino_softmaxed_centered_list
            else:
                # 如果是张量，分拆成每个全局裁剪对应的部分
                for i in range(n_global_crops):
                    teacher_chunks.append(teacher_dino_softmaxed_centered_list[i:i+1])
            
            # 确保维度匹配
            min_len = min(len(student_local_chunks), len(teacher_chunks))
            if min_len < len(student_local_chunks) or min_len < len(teacher_chunks):
                logger.warning(f"调整学生局部裁剪组数量从{len(student_local_chunks)}到{min_len}，教师输出从{len(teacher_chunks)}到{min_len}")
            
            student_local_chunks = student_local_chunks[:min_len]
            teacher_chunks = teacher_chunks[:min_len]
            
            # 计算局部裁剪损失
            if student_local_chunks and teacher_chunks:
                dino_local_crops_loss = self.dino_loss(
                    student_output_list=student_local_chunks,
                    teacher_out_softmaxed_centered_list=teacher_chunks,
                ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
                
                # 存储用于显示
                loss_dict["dino_local_crops_loss"] = dino_local_crops_loss.detach().clone()
                
                # 累积损失 (为梯度累积正确缩放)
                loss_accumulator += self.dino_loss_weight * dino_local_crops_loss / grad_accumulation_steps
        
        # 计算全局裁剪损失
        if self.do_dino:
            if isinstance(teacher_dino_softmaxed_centered_list, list):
                # 如果是列表，将所有元素连接成一个张量
                teacher_tensor = torch.cat(teacher_dino_softmaxed_centered_list, dim=0)
            else:
                # 如果已经是张量，使用原样
                teacher_tensor = teacher_dino_softmaxed_centered_list
            
            # 计算全局裁剪损失
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[teacher_tensor],
                )
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )
            
            # 存储用于显示
            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss.detach().clone()
            
            # 累积损失 (为梯度累积正确缩放)
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss / grad_accumulation_steps
            
            # 应用KoLeo损失 (如果启用)
            if self.do_koleo:
                koleo_weight = self.cfg.dino.koleo_loss_weight * 0.01  # 降低权重
                
                koleo_values = []
                for p in global_batches:
                    # 检查输入
                    p_safe = check_tensor(p, "KoLeo输入")
                    
                    try:
                        # 直接计算KoLeo损失 (不使用torch.no_grad)
                        val = self.koleo_loss(p_safe)
                        # 裁剪异常值
                        val = torch.clamp(val, -1.0, 1.0)
                        koleo_values.append(val)
                    except Exception as e:
                        logger.error(f"KoLeo损失计算错误: {str(e)}")
                        # 使用小的默认值
                        koleo_values.append(torch.tensor(0.1).to(p.device))
                
                # 安全求和
                if koleo_values:
                    koleo_loss = sum(koleo_values) / len(koleo_values)
                    # 存储用于显示
                    loss_dict["koleo_loss"] = (koleo_weight * koleo_loss).detach().clone()
                    loss_dict["koleo_loss_raw"] = koleo_loss.detach().clone()
                    
                    # 累积损失 (为梯度累积正确缩放)
                    loss_accumulator += koleo_weight * koleo_loss / grad_accumulation_steps
        
        # 检查总损失是否合理
        total_loss = loss_accumulator.clone().detach()
        if total_loss.isnan() or total_loss.isinf() or total_loss > 100.0:
            logger.warning(f"检测到异常总损失: {total_loss}，将进行裁剪")
            # 裁剪损失
            loss_accumulator = torch.clamp(loss_accumulator, -10.0, 10.0)
        
        # 执行反向传播 (但不更新权重，由外层训练循环负责)
        self.backprop_loss(loss_accumulator)
        
        # 返回损失信息
        return loss_dict

    @torch.no_grad()
    def update_teacher(self, m):
        """更新教师模型（EMA更新）- 支持分布式环境"""
        # 检查m值是否合理
        if m >= 0.9999:  # 避免动量太接近1
            m = 0.9999
            
        with torch.no_grad():
            # 更新所有组件，包括backbone
            for teacher_key, student_key in zip(self.teacher.keys(), self.student.keys()):
                teacher_module = self.teacher[teacher_key]
                student_module = self.student[student_key]
                
                # 在分布式环境中，需要获取DDP包装的模块
                if hasattr(student_module, 'module'):
                    student_module_unwrapped = student_module.module
                else:
                    student_module_unwrapped = student_module
                
                # 根据组件类型区分更新策略
                if teacher_key == "dino_head":
                    # DINO头部完全更新
                    for teacher_param, student_param in zip(teacher_module.parameters(), 
                                                        student_module_unwrapped.parameters()):
                        teacher_param.data.mul_(m).add_(student_param.data, alpha=1-m)
                elif teacher_key == "backbone":
                    # 对于backbone，选择性更新非LoRA层
                    for name, teacher_param in teacher_module.named_parameters():
                        # 跳过LoRA参数，只更新原始模型参数
                        if "lora_" not in name and name in dict(student_module_unwrapped.named_parameters()):
                            student_param = dict(student_module_unwrapped.named_parameters())[name]
                            # 使用较小的更新率
                            teacher_param.data.mul_(m).add_(student_param.data, alpha=1-m)

    def train(self):
        """设置模型为训练模式"""
        super().train()
        self.teacher.eval()  # 教师模型始终为评估模式

    def get_params_groups(self):
        """获取参数组，用于优化器"""
        all_params_groups = []
        for key, module in self.student.items():
            # 对于DDP包装的模块，获取内部模块
            if hasattr(module, 'module'):
                module_unwrapped = module.module
            else:
                module_unwrapped = module
            
            all_params_groups += self._get_params_for_module(module_unwrapped)
        return all_params_groups
    
    def _get_params_for_module(self, module):
        """获取模块的参数组，处理LoRA参数"""
        params_groups = []
        # 获取所有参数
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
                
            # 默认参数组
            param_group = {
                "params": param, 
                "is_last_layer": "last_layer" in name,
                "lr_multiplier": 1.0, 
                "wd_multiplier": 1.0,
                "name": name
            }
            
            # 特殊处理
            if "lora_" in name:
                # LoRA参数使用较大的学习率
                param_group["lr_multiplier"] = 5.0  # 提高乘数增加训练速度
                
            if name.endswith(".bias") or "norm" in name or "gamma" in name:
                param_group["wd_multiplier"] = 0.0
                
            params_groups.append(param_group)
            
        return params_groups

    def prepare_for_distributed_training(self):
        """为分布式训练准备模型"""
        logger.info("准备模型(支持分布式训练模式)")
        
        # 对于教师模型我们不从学生模型加载，而是使用保存的原始权重
        if self.original_state_dict:
            for k, v in self.original_state_dict.items():
                if k in self.teacher:
                    self.teacher[k].load_state_dict(v)
            
            logger.info("教师模型已从原始权重初始化")
        else:
            logger.warning("没有找到原始权重，教师模型可能未正确初始化")
        
        # 确保所有进程的教师模型一致
        if dist.is_initialized():
            for k, v in self.teacher.items():
                for param in v.parameters():
                    dist.broadcast(param.data, 0)
            logger.info("教师模型已在所有进程间广播同步")
    
    def _init_stability_measures(self):
        """初始化所有稳定性措施"""
        logger.info("初始化稳定性措施")
        self._reinit_normalization_params()
        self._reinit_lora_params()

    def _reinit_normalization_params(self):
        """重新初始化所有层规范化参数，提高训练稳定性"""
        logger.info("重新初始化层规范化参数")
        
        # 保持较小的初始值，防止大梯度
        for key, module in self.student.items():
            for name, sub_module in module.named_modules():
                # 初始化所有层规范化模块
                if 'norm' in name.lower() or 'ln' in name.lower():
                    if hasattr(sub_module, 'weight') and sub_module.weight is not None:
                        # 接近1但有微小的扰动
                        torch.nn.init.normal_(sub_module.weight, mean=1.0, std=0.01)
                    if hasattr(sub_module, 'bias') and sub_module.bias is not None:
                        torch.nn.init.constant_(sub_module.bias, 0)
                
                # 初始化层缩放参数
                if 'ls1' in name.lower() or 'ls2' in name.lower():
                    if hasattr(sub_module, 'gamma') and sub_module.gamma is not None:
                        # 从较小的值开始
                        with torch.no_grad():
                            sub_module.gamma.fill_(0.1)

    def _reinit_lora_params(self):
        """重新初始化所有LoRA参数，使用更合理的初始化策略"""
        logger.info("重新初始化LoRA参数")
        
        for key, module in self.student.items():
            for name, param in module.named_parameters():
                if 'lora_A' in name:
                    # 使用kaiming初始化A矩阵
                    torch.nn.init.kaiming_normal_(param, mode='fan_out')
                elif 'lora_B' in name:
                    # B矩阵使用小但非零的初始化
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)