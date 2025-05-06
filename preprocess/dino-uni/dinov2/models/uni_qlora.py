"""Q-LoRA（量化低秩适配器）实现，用于UNI模型的高效微调"""

import logging
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import re

logger = logging.getLogger("dinov2")

def find_all_linear_names(model):
    """找出模型中所有线性层的名称
    
    Args:
        model: PyTorch模型
        
    Returns:
        线性层名称列表
    """
    linear_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(name)
            
    return linear_layers

def find_attention_module_names(model):
    """找出模型中的注意力模块名称
    
    Args:
        model: PyTorch模型
        
    Returns:
        注意力模块名称列表
    """
    attention_modules = []
    
    for name, module in model.named_modules():
        # ViT注意力模块通常包含query, key, value和proj
        if any(x in name for x in ['attn', 'attention']):
            attention_modules.append(name)
        elif any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'out_proj']):
            attention_modules.append(name)
            
    return attention_modules

def get_target_modules(model, target_modules_type="all_linear"):
    """获取目标模块列表
    
    Args:
        model: PyTorch模型
        target_modules_type: 目标模块类型，可以是"all_linear"或"attention_only"或自定义列表
        
    Returns:
        目标模块名称列表
    """
    if isinstance(target_modules_type, list):
        return target_modules_type
    
    if target_modules_type == "all_linear":
        return find_all_linear_names(model)
    elif target_modules_type == "attention_only":
        return find_attention_module_names(model)
    else:
        logger.warning(f"未知的目标模块类型: {target_modules_type}，使用默认的注意力模块")
        return find_attention_module_names(model)

def apply_qlora_to_model(model, config=None):
    """为模型应用Q-LoRA
    
    Args:
        model: UNIEncoder模型
        config: LoRA配置，如果为None则使用默认配置
        
    Returns:
        应用了LoRA的模型
    """
    if not hasattr(model, 'model'):
        logger.error("模型必须有一个'model'属性，指向底层模型实现")
        return model
    
    base_model = model.model
    
    # 准备模型进行kbit训练
    base_model = prepare_model_for_kbit_training(base_model)
    
    # 如果没有提供配置，使用默认配置
    if config is None:
        target_modules = get_target_modules(base_model, "attention_only")
        
        config = LoraConfig(
            r=16,                          # LoRA矩阵的秩
            lora_alpha=32,                 # LoRA缩放因子
            target_modules=target_modules, # 要应用LoRA的模块
            lora_dropout=0.1,              # LoRA dropout
            bias="none",                   # 不为偏置参数使用LoRA
            task_type=TaskType.FEATURE_EXTRACTION  # 任务类型
        )
    
    # 应用LoRA
    logger.info(f"将LoRA应用到以下模块: {config.target_modules}")
    peft_model = get_peft_model(base_model, config)
    
    # 更新原始模型中的model属性
    model.model = peft_model
    
    # 添加辅助方法，以便外部查询可训练参数
    model.print_trainable_parameters = peft_model.print_trainable_parameters
    
    return model

def create_qlora_config(cfg):
    """基于配置创建LoRA配置
    
    Args:
        cfg: 配置对象
        
    Returns:
        LoraConfig对象
    """
    # 获取LoRA参数
    r = getattr(cfg.student, 'lora_r', 16)
    alpha = getattr(cfg.student, 'lora_alpha', 32)
    dropout = getattr(cfg.student, 'lora_dropout', 0.1)
    
    if hasattr(cfg.student, 'lora_target_modules'):
        if isinstance(cfg.student.lora_target_modules, list):
            target_modules = cfg.student.lora_target_modules
        else:
            target_modules = cfg.student.lora_target_modules
    else:
        target_modules = "attention_only"
    
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

def get_lora_parameters(model):
    """获取模型中的LoRA参数
    
    Args:
        model: 应用了LoRA的模型
        
    Returns:
        LoRA参数列表
    """
    lora_params = []
    
    # 检查是否为UNIEncoder模型
    if hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
    
    # 检查是否有peft_config属性（表明使用了PEFT库）
    if hasattr(base_model, 'peft_config'):
        # 收集LoRA参数
        for name, param in base_model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_params.append(param)
    
    return lora_params

def count_parameters(model):
    """计算模型的参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        total_params: 总参数数量
        trainable_params: 可训练参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def print_qlora_model_info(model):
    """打印应用了Q-LoRA的模型信息
    
    Args:
        model: 应用了Q-LoRA的模型
    """
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    else:
        total_params, trainable_params = count_parameters(model)
        logger.info(f"总参数量: {total_params:,}")
        logger.info(f"可训练参数量: {trainable_params:,}")
        logger.info(f"可训练参数占比: {100 * trainable_params / total_params:.4f}%")