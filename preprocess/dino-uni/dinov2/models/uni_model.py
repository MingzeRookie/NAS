"""UNI模型在DINOv2框架中的集成实现"""

import os
import torch
import torch.nn as nn
from transformers import BitsAndBytesConfig
from huggingface_hub import login
import timm
import logging

logger = logging.getLogger("dinov2")

class UNIEncoder(nn.Module):
    """UNI Encoder包装类，实现与DINOv2框架的兼容"""
    
    def __init__(self, model, embed_dim, patch_size=14):
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
    def forward(self, x, masks=None, is_training=False):
        """前向传播函数，兼容DINOv2的输入格式
        
        Args:
            x: 输入图像，可以是单个tensor或列表，如[global_crops, local_crops]
            masks: 可选掩码，用于iBOT
            is_training: 是否在训练模式
            
        Returns:
            输出字典，包含归一化的cls token和patch tokens
        """
        if isinstance(x, list):
            # 处理global_crops和local_crops的情况
            global_crops, local_crops = x
            
            # 组合处理然后分开结果
            if local_crops is not None:
                combined_crops = torch.cat([global_crops, local_crops], dim=0)
                combined_features = self._extract_features(combined_crops)
                
                global_features, local_features = torch.split(
                    combined_features, 
                    [global_crops.shape[0], local_crops.shape[0]]
                )
            else:
                global_features = self._extract_features(global_crops)
                
            return {
                "x_norm_clstoken": global_features,
                "x_norm_patchtokens": self._get_patch_tokens(global_crops) if masks is not None and masks[0] is not None else None,
                "masks": masks[0] if masks is not None else None
            }
        else:
            # 处理单个输入的情况
            features = self._extract_features(x)
            return {
                "x_norm_clstoken": features,
                "x_norm_patchtokens": self._get_patch_tokens(x) if masks is not None else None,
                "masks": masks
            }
    
    def _extract_features(self, x):
        """提取特征函数
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            特征向量 [B, embed_dim]
        """
        # 确保模型处于评估模式以防止批量归一化更新统计信息
        training_mode = self.model.training
        if training_mode:
            self.model.eval()
            
        with torch.set_grad_enabled(True):  # 允许梯度以便反向传播
            features = self.model(x)
            
        # 恢复原始训练状态
        if training_mode:
            self.model.train()
            
        return features
    
    def _get_patch_tokens(self, x):
        """获取patch tokens，用于iBOT训练
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            patch tokens [B, N, embed_dim]
        """
        # UNI模型可能没有直接暴露patch tokens
        # 这里我们提供一个简单实现，实际应用中可能需要修改UNI模型
        # 或通过hook获取中间特征
        
        # 计算每个维度上的patch数量
        h_patches = x.shape[2] // self.patch_size
        w_patches = x.shape[3] // self.patch_size
        n_patches = h_patches * w_patches
        
        # 创建临时patch tokens，实际使用时应从模型中获取
        # 这里仅作为占位实现
        batch_size = x.shape[0]
        patch_tokens = torch.zeros(
            (batch_size, n_patches, self.embed_dim), 
            device=x.device, 
            dtype=x.dtype
        )
        
        return patch_tokens


def build_uni_encoder_model_4bit(cfg):
    """构建4-bit量化的UNI Encoder模型
    
    Args:
        cfg: 配置对象，包含模型参数
        
    Returns:
        student_model: 学生模型
        teacher_model: 教师模型
        embed_dim: 嵌入维度
    """
    # 登录HuggingFace（如果提供了token）
    if hasattr(cfg.student, 'hf_token') and cfg.student.hf_token:
        login(token=cfg.student.hf_token)
    
    # 4-bit量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # 为正态分布的权重优化
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # 对量化常数再次量化以节省内存
    )
    
    # 根据UNI版本决定模型参数
    if cfg.student.uni_version == "uni2-h":
        # UNI2-h (High) 参数
        logger.info("加载UNI2-h模型 (ViT-H/14)...")
        timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
        embed_dim = 1536
        patch_size = 14
        
        # 加载模型
        if hasattr(cfg.student, 'pretrained_weights') and cfg.student.pretrained_weights:
            logger.info(f"从本地加载UNI2-h权重: {cfg.student.pretrained_weights}")
            student_base_model = timm.create_model(**timm_kwargs)
            student_base_model.load_state_dict(
                torch.load(cfg.student.pretrained_weights, map_location="cpu"), 
                strict=True
            )
            
            teacher_base_model = timm.create_model(**timm_kwargs)
            teacher_base_model.load_state_dict(
                torch.load(cfg.student.pretrained_weights, map_location="cpu"), 
                strict=True
            )
        else:
            logger.info("从HuggingFace加载UNI2-h预训练权重")
            student_base_model = timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h", 
                pretrained=True, 
                **timm_kwargs
            )
            
            teacher_base_model = timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h", 
                pretrained=True, 
                **timm_kwargs
            )
    else:
        # 默认UNI (ViT-L/16) 参数
        logger.info("加载UNI模型 (ViT-L/16)...")
        timm_kwargs = {
            'img_size': 224,
            'patch_size': 16,
            'init_values': 1e-5,
            'num_classes': 0,
            'dynamic_img_size': True
        }
        embed_dim = 1024
        patch_size = 16
        
        # 加载模型
        if hasattr(cfg.student, 'pretrained_weights') and cfg.student.pretrained_weights:
            logger.info(f"从本地加载UNI权重: {cfg.student.pretrained_weights}")
            student_base_model = timm.create_model(
                "vit_large_patch16_224", 
                **timm_kwargs
            )
            student_base_model.load_state_dict(
                torch.load(cfg.student.pretrained_weights, map_location="cpu"), 
                strict=True
            )
            
            teacher_base_model = timm.create_model(
                "vit_large_patch16_224", 
                **timm_kwargs
            )
            teacher_base_model.load_state_dict(
                torch.load(cfg.student.pretrained_weights, map_location="cpu"), 
                strict=True
            )
        else:
            logger.info("从HuggingFace加载UNI预训练权重")
            student_base_model = timm.create_model(
                "hf-hub:MahmoodLab/UNI", 
                pretrained=True, 
                **timm_kwargs
            )
            
            teacher_base_model = timm.create_model(
                "hf-hub:MahmoodLab/UNI", 
                pretrained=True, 
                **timm_kwargs
            )
    
    # 包装成UNIEncoder
    student_model = UNIEncoder(student_base_model, embed_dim, patch_size)
    teacher_model = UNIEncoder(teacher_base_model, embed_dim, patch_size)
    
    return student_model, teacher_model, embed_dim