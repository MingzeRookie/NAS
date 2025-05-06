# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 您的组织名称
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

logger = logging.getLogger("dinov2_qlora")

class LoRALayer(nn.Module):
    """LoRA层的基础实现"""
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        # 缩放因子
        self.scaling = self.lora_alpha / self.r
        # dropout层
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        # 是否合并权重
        self.merge_weights = merge_weights

    def reset_parameters(self):
        raise NotImplementedError

class LoRALinear(LoRALayer):
    """应用到线性层上的LoRA"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        bias: bool = False,
    ):
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.fan_in_fan_out = fan_in_fan_out
        # 投影到低维空间的权重
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        # 从低维空间投影回高维空间的权重
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        # 可选的偏置
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
                
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化LoRA的权重参数"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, x: torch.Tensor):
        # dropout应用于输入
        x_dropout = self.lora_dropout(x)
        # 低秩更新
        lora_output = (x_dropout @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return lora_output

class QuantizedLoRALinear(nn.Module):
    """量化+LoRA的线性层"""
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        bias_type: str = "none",  # "none", "all", "lora_only"
        bits: int = 4,
        group_size: int = 128,
        double_quant: bool = True,
    ):
        super().__init__()
        self.base_layer = base_layer
        # 创建LoRA适配器
        self.lora_adapter = LoRALinear(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias_type == "lora_only",
            merge_weights=False,
        )
        
        # 配置
        self.bits = bits
        self.group_size = group_size
        self.double_quant = double_quant
        
        # 保存原始权重的副本，并进行量化
        if not hasattr(self.base_layer, "weight_orig"):
            self.base_layer.register_buffer("weight_orig", self.base_layer.weight.data.clone())
            
        # 设置基础层权重需要梯度为False（冻结）
        self.base_layer.weight.requires_grad_(False)
        
        # 处理偏置
        if bias_type == "all" and self.base_layer.bias is None:
            self.base_layer.bias = nn.Parameter(torch.zeros(base_layer.out_features))
        elif bias_type == "none" and self.base_layer.bias is not None:
            self.base_layer.bias = None
            
        # 简化的量化 - 实际上我们只是克隆权重而不进行量化
        # 对于真实场景，应该使用更复杂的量化方法
        self.base_layer.weight.data.copy_(self.base_layer.weight_orig.data)
                
    def forward(self, x: torch.Tensor):
        # 原始实现 - 适用于小批量输入
        if x.shape[0] * x.shape[1] < 500000:  # 适当的阈值
            base_output = self.base_layer(x)
            lora_output = self.lora_adapter(x)
            return base_output + lora_output
        
        # 大批量输入使用分块处理
        chunk_size = 4  # 每次处理的批次数量
        outputs = []
        
        # 分批处理
        for i in range(0, x.shape[0], chunk_size):
            # 提取当前块
            chunk = x[i:i+chunk_size]
            
            # 计算基础层输出
            base_output = self.base_layer(chunk)
            
            # 计算LoRA输出
            if self.training and self.lora_adapter.lora_dropout is not None:
                chunk_dropout = self.lora_adapter.lora_dropout(chunk)
            else:
                chunk_dropout = chunk
            
            # 手动计算LoRA输出，避免创建太大的中间张量
            lora_output = (chunk_dropout @ self.lora_adapter.lora_A.T @ self.lora_adapter.lora_B.T) * self.lora_adapter.scaling
            
            # 合并输出并添加到结果列表
            outputs.append(base_output + lora_output)
        
        # 合并所有分块结果
        return torch.cat(outputs, dim=0)

def get_modules_by_name(model, target_modules):
    """获取模型中与目标名称匹配的模块"""
    module_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for target in target_modules:
                if target in name:
                    module_dict[name] = module
                    break
    
    return module_dict

def apply_qlora(
    model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: List[str] = ["qkv", "fc1", "fc2"],
    bias: str = "none",
    bits: int = 4,
    group_size: int = 128,
    double_quant: bool = True,
) -> int:
    """
    将Q-LoRA应用到模型上
    
    Args:
        model: 要应用Q-LoRA的模型
        r: LoRA的秩
        alpha: LoRA的alpha参数
        dropout: LoRA的dropout率
        target_modules: 要应用LoRA的模块名称列表
        bias: 偏置处理方式，"none"、"all"或"lora_only"
        bits: 量化位数
        group_size: 量化组大小
        double_quant: 是否使用double量化
        
    Returns:
        转换的层数
    """
    # 获取所有符合条件的模块
    modules_to_convert = get_modules_by_name(model, target_modules)
    converted_layers = 0
    
    # 直接替换模块
    for name, module in modules_to_convert.items():
        parent_name, child_name = name.rsplit(".", 1)
        
        # 获取父模块
        parent = model
        for part in parent_name.split("."):
            if hasattr(parent, part):
                parent = getattr(parent, part)
            else:
                logger.warning(f"找不到模块: {part} in {parent_name}")
                continue
        
        # 替换模块
        try:
            qlora_module = QuantizedLoRALinear(
                module,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias_type=bias,
                bits=bits,
                group_size=group_size,
                double_quant=double_quant,
            )
            setattr(parent, child_name, qlora_module)
            converted_layers += 1
            logger.info(f"成功转换层: {name}, 输入维度: {module.in_features}, 输出维度: {module.out_features}")
        except Exception as e:
            logger.error(f"转换层 {name} 失败: {e}")
    
    logger.info(f"总共转换了 {converted_layers} 个层")
    return converted_layers

def get_trainable_params(model):
    """获取模型中可训练的参数"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return trainable_params, all_param, round(trainable_params / all_param * 100, 3)
