"""DINOv2模型构建模块，支持UNI模型集成"""

import logging

from . import vision_transformer as vits
from .uni_model import build_uni_encoder_model_4bit
from .uni_qlora import apply_qlora_to_model, create_qlora_config, print_qlora_model_info


logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224):
    """构建模型函数（原DINOv2实现）
    
    Args:
        args: 参数对象
        only_teacher: 是否只构建教师模型
        img_size: 图像尺寸
        
    Returns:
        student: 学生模型
        teacher: 教师模型
        embed_dim: 嵌入维度
    """
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
        )
        return student, teacher, student.embed_dim
    else:
        raise ValueError(f"不支持的架构: {args.arch}")


def build_model_from_cfg(cfg, only_teacher=False):
    """从配置构建模型
    
    Args:
        cfg: 配置对象
        only_teacher: 是否只构建教师模型
        
    Returns:
        student_backbone: 学生模型主干
        teacher_backbone: 教师模型主干
        embed_dim: 嵌入维度
    """
    # 检查是否为UNI模型
    if hasattr(cfg.student, 'arch') and cfg.student.arch == "uni_encoder":
        logger.info("构建UNI Encoder模型")
        
        # 构建基本UNI模型
        student_backbone, teacher_backbone, embed_dim = build_uni_encoder_model_4bit(cfg)
        
        # 如果配置了使用LoRA/Q-LoRA，则应用适配器
        if hasattr(cfg.student, 'use_lora') and cfg.student.use_lora:
            logger.info("为学生模型应用Q-LoRA适配器")
            
            # 创建LoRA配置
            lora_config = create_qlora_config(cfg)
            
            # 为学生模型应用LoRA
            student_backbone = apply_qlora_to_model(student_backbone, lora_config)
            
            # 打印Q-LoRA模型信息
            print_qlora_model_info(student_backbone)
        
        if only_teacher:
            return teacher_backbone, embed_dim
            
        return student_backbone, teacher_backbone, embed_dim
    else:
        # 使用原始的模型构建函数
        return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)