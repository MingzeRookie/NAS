#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 您的组织名称
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf

from models.uni_ssl_meta_arch import UNISSLMetaArch
from models.lora_adapter import get_trainable_params
from data.patch_dataset import create_patch_dataset
from data_transforms import MultiCropTransform, collate_multi_crop_batch

import logging
from dinov2.utils.utils import fix_random_seeds, CosineScheduler
torch.set_printoptions(precision=10)
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("dinov2_qlora")


def get_args_parser():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser('UNI-V2 Q-LoRA训练', add_help=True)
    parser.add_argument('--config', default='configs/qlora_config.yaml', type=str, help='配置文件路径')
    parser.add_argument('--output_dir', default='./output', type=str, help='输出目录')
    parser.add_argument('--seed', default=0, type=int, help='随机种子')
    parser.add_argument('--num_workers', default=10, type=int, help='数据加载器的worker数量')
    parser.add_argument('--batch_size', default=32, type=int, help='每个GPU的批次大小')
    parser.add_argument('--epochs', default=100, type=int, help='训练的总轮数')
    parser.add_argument('--save_freq', default=10, type=int, help='保存检查点的频率')
    parser.add_argument('--eval_freq', default=5, type=int, help='评估的频率')
    parser.add_argument('--resume', default='', type=str, help='从检查点恢复')
    # 新增分布式训练参数
    parser.add_argument('--distributed', action='store_true', help='是否使用分布式训练')
    parser.add_argument('--local-rank', '--local_rank', dest='local_rank', type=int, default=-1, help='分布式训练的本地排名')
    parser.add_argument('--dist_url', default='env://', type=str, help='分布式训练的URL')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='分布式训练的后端')
    return parser


def init_distributed_mode(args):
    """
    初始化分布式训练环境
    """
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        else:
            logger.info('未找到环境变量，使用命令行参数进行分布式训练')
            # 处理环境变量中的LOCAL_RANK (torchrun自动设置)
            if 'LOCAL_RANK' in os.environ:
                args.local_rank = int(os.environ['LOCAL_RANK'])
            args.rank = args.local_rank
            args.gpu = args.local_rank
            args.world_size = torch.cuda.device_count()
            os.environ['WORLD_SIZE'] = str(args.world_size)
        
        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'
        logger.info(f'| 分布式初始化 (rank {args.rank}): {args.dist_url}')
        dist.init_process_group(
            backend=args.dist_backend, 
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
        dist.barrier()
        setup_for_distributed(args.rank == 0)
    else:
        args.gpu = 0
        args.rank = 0
        args.world_size = 1
        setup_for_distributed(True)


def setup_for_distributed(is_master):
    """
    配置分布式环境的日志设置
    """
    # 只在主进程上输出信息
    if not is_master:
        import builtins as __builtin__
        builtin_print = __builtin__.print
        
        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if force:
                builtin_print(*args, **kwargs)
        
        __builtin__.print = print


def load_config(config_path):
    """
    加载配置文件
    """
    logger.info(f"从{config_path}加载配置")
    return OmegaConf.load(config_path)


def save_config(config, output_dir):
    """
    保存配置文件
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(config=config, f=f)
    logger.info(f"配置已保存到{config_path}")


def build_model(cfg):
    """
    构建UNI-V2 Q-LoRA模型
    """
    logger.info("构建UNI-V2 Q-LoRA模型")
    model = UNISSLMetaArch(cfg).cuda()
    return model


def build_dataloader(cfg, args, transform=None):
    """
    构建数据加载器(支持分布式)
    """
    # 创建数据集
    if transform is None:
        transform = MultiCropTransform(
            global_crops_scale=tuple(cfg.crops.global_crops_scale),
            local_crops_scale=tuple(cfg.crops.local_crops_scale),
            local_crops_number=cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
        )
    
    # 确定缓存文件路径和使用缓存的设置
    cache_file = None
    use_cache = hasattr(cfg.train, 'cache_dataset') and cfg.train.cache_dataset
    
    if use_cache:
        if hasattr(cfg.train, 'dataset_cache_path'):
            cache_file = cfg.train.dataset_cache_path
        else:
            cache_file = os.path.join(cfg.train.output_dir, 'dataset_cache.pt')
    
    dataset = create_patch_dataset(
        root_dir=cfg.train.dataset_path,
        transform=transform,
        min_patches_per_class=10,
        max_patches_per_class=100,
        patch_size=cfg.crops.global_crops_size,
        cache_file=cache_file,
        use_cache=use_cache,
    )
    
    # 创建分布式采样器
    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False  # 使用采样器时不需要shuffle
    else:
        sampler = None
        shuffle = True
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        collate_fn=collate_multi_crop_batch,
    )
    
    return dataloader, sampler


def build_optimizer(cfg, model):
    """
    构建优化器
    """
    params_groups = model.get_params_groups()
    optimizer = torch.optim.AdamW(
        params_groups,
        weight_decay=cfg.optim.weight_decay,
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
    )
    return optimizer


def build_schedulers(cfg):
    """
    构建调度器 - 全面修复版本
    """
    # 定义周期长度
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    
    # 修复学习率调度 - 使用更保守的范围
    lr = dict(
        base_value=cfg.optim.base_lr,
        final_value=max(cfg.optim.min_lr, 0.00001),  # 确保最小学习率不会太小
        total_iters=cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim.warmup_epochs * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0.00001,  # 从一个合理的小值开始
    )
    
    # 修复权重衰减调度 - 固定不变
    wd = dict(
        base_value=cfg.optim.weight_decay,
        final_value=cfg.optim.weight_decay,  # 保持权重衰减不变，避免增长
        total_iters=cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH,
    )
    
    # 修复动量调度 - 设置最大值并限制增长率
    momentum = dict(
        base_value=cfg.teacher.momentum_teacher,
        final_value=min(cfg.teacher.final_momentum_teacher, 0.9995),  # 限制更严格
        total_iters=cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH,
    )
    
    # 教师温度调度
    teacher_temp = dict(
        base_value=cfg.teacher.teacher_temp,
        final_value=cfg.teacher.teacher_temp,
        total_iters=cfg.teacher.warmup_teacher_temp_epochs * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher.warmup_teacher_temp_epochs * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher.warmup_teacher_temp,
    )
    
    # 创建余弦调度器
    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)
    
    # 检查调度器值域是否合理
    logger.info(f"学习率范围: {lr['base_value']} -> {lr['final_value']}")
    logger.info(f"权重衰减范围: {wd['base_value']} -> {wd['final_value']}")
    logger.info(f"动量范围: {momentum['base_value']} -> {momentum['final_value']}")
    logger.info(f"教师温度范围: {teacher_temp['start_warmup_value']} -> {teacher_temp['base_value']}")
    
    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    """
    应用优化器调度
    """
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def save_checkpoint(model, optimizer, epoch, iteration, cfg, filename="checkpoint.pth"):
    """
    保存检查点
    """
    output_dir = Path(cfg.train.output_dir)
    save_path = output_dir / filename
    
    # 处理DDP包装的模型
    model_state_dict = {}
    for k, v in model.student.items():
        if isinstance(v, DDP):
            model_state_dict[k] = v.module.state_dict()
        else:
            model_state_dict[k] = v.state_dict()
    
    checkpoint = {
        "model": model_state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "iteration": iteration,
        "cfg": OmegaConf.to_container(cfg),
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"检查点已保存到{save_path}")


def resume_from_checkpoint(model, optimizer, resume_path, args):
    """
    从检查点恢复
    """
    logger.info(f"从{resume_path}恢复")
    checkpoint = torch.load(resume_path, map_location="cpu")
    
    # 加载模型权重
    for k, v in checkpoint["model"].items():
        if k in model.student:
            module = model.student[k]
            # 处理DDP包装的模型
            if args.distributed and isinstance(module, DDP):
                msg = module.module.load_state_dict(v)
            else:
                msg = module.load_state_dict(v)
            logger.info(f"模块 {k} 加载结果: {msg}")
    
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint["epoch"], checkpoint["iteration"]


def monitor_gradients(model, iteration, freq=200):
    """监控梯度状态，仅在关键节点输出信息"""
    if iteration % freq != 0:
        return
    
    # 检查梯度状态
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            # 计算梯度统计信息
            grad_stats[name] = {
                "mean": grad.mean().item(),
                "std": grad.std().item(),
                "min": grad.min().item(),
                "max": grad.max().item(),
                "has_nan": torch.isnan(grad).any().item(),
                "has_inf": torch.isinf(grad).any().item(),
            }
    
    # 只记录异常梯度信息，减少日志量
    has_issues = False
    issues_info = []
    
    for name, stats in grad_stats.items():
        # 只在有问题时记录
        if stats["has_nan"] or stats["has_inf"] or abs(stats["mean"]) > 1.0 or stats["max"] > 10.0:
            has_issues = True
            issues_info.append(f"  - {name}: 异常梯度! 均值={stats['mean']:.6f}, 最大值={stats['max']:.6f}")
        
        # 只对LoRA参数进行定期抽样记录，且减少频率
        elif "lora_" in name and iteration % (freq * 5) == 0 and iteration > 0:
            # 每5个监控周期才记录一次LoRA参数，并且只记录少量参数
            if hash(name) % 20 == 0:  # 使用简单hash只保留约5%的参数记录
                issues_info.append(f"  - {name}: 均值={stats['mean']:.6f}, 范围=[{stats['min']:.6f}, {stats['max']:.6f}]")
    
    # 只在存在问题或关键迭代点记录
    if has_issues:
        logger.warning(f"迭代 {iteration} 的梯度存在异常:")
        for info in issues_info:
            logger.warning(info)
    elif iteration % (freq * 10) == 0 and iteration > 0:
        # 每10个监控周期输出一次简要状态
        logger.info(f"迭代 {iteration} 梯度状态正常")


def main():
    # 解析命令行参数
    args = get_args_parser().parse_args()
    
    # 初始化分布式环境
    init_distributed_mode(args)
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 更新配置
    cfg.train.batch_size_per_gpu = args.batch_size
    cfg.optim.warmup_epochs = 10
    cfg.optim.min_lr = 0.00005
    cfg.optim.base_lr = 0.0001
    cfg.train.output_dir = args.output_dir
    cfg.train.seed = args.seed
    cfg.train.num_workers = args.num_workers
    cfg.optim.epochs = args.epochs
    cfg.train.saveckp_freq = args.save_freq
    cfg.evaluation.eval_period_iterations = args.eval_freq * cfg.train.OFFICIAL_EPOCH_LENGTH
    
    # 保存配置 (只在主进程上)
    if args.rank == 0:
        save_config(cfg, args.output_dir)
    
    # 设置随机种子 (每个进程使用不同的种子)
    fix_random_seeds(cfg.train.seed + args.rank)
    
    # 构建模型
    model = build_model(cfg)
    model.train()
    
    # 包装模型用于DDP (只在需要分布式的情况下)
    if args.distributed:
        # 找到所有需要DDP包装的子模块
        for key in list(model.student.keys()):
            module = model.student[key]
            # 将子模块包装到DDP中
            model.student[key] = DDP(
                module, 
                device_ids=[args.gpu],
                output_device=args.gpu,
                broadcast_buffers=False,
                find_unused_parameters=False
            )
        
        logger.info(f"进程 {args.rank}: 模型已包装到DDP中")
    
    # 准备模型用于分布式训练
    model.prepare_for_distributed_training()
    
    # 显示可训练参数信息 (只在主进程)
    if args.rank == 0:
        for key, module in model.student.items():
            module_to_check = module.module if args.distributed else module
            trainable, total, percentage = get_trainable_params(module_to_check)
            logger.info(f"{key} - 可训练参数: {trainable:,d} ({percentage}%) / 总参数: {total:,d}")
    
    # 构建数据加载器
    dataloader, sampler = build_dataloader(cfg, args)
    
    # 构建优化器和调度器
    optimizer = build_optimizer(cfg, model)
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)
    
    # 准备检查点
    start_epoch = 0
    iteration = 0
    
    # 恢复训练（如果需要）
    if args.resume:
        start_epoch, iteration = resume_from_checkpoint(model, optimizer, args.resume, args)
    
    # 训练循环
    logger.info("开始训练！")
    max_iter = cfg.optim.epochs * len(dataloader)
    end = time.time()
    grad_accumulation_steps = 4  # 使用梯度累积
    logger.info(f"使用梯度累积: {grad_accumulation_steps}步")
    for epoch in range(start_epoch, cfg.optim.epochs):
        logger.info(f"开始第 {epoch+1}/{cfg.optim.epochs} 轮训练")
        
        # 设置分布式采样器的epoch
        if args.distributed:
            sampler.set_epoch(epoch)
        
        # 重置数据加载器
        data_loader_iterator = iter(dataloader)
        
        for i in range(len(dataloader)):
            # 更新迭代计数
            iteration = epoch * len(dataloader) + i
            
            # 获取下一批数据
            try:
                data = next(data_loader_iterator)
            except StopIteration:
                data_loader_iterator = iter(dataloader)
                data = next(data_loader_iterator)
            
            # 计算当前进度
            progress = iteration / max_iter
            
            # 应用学习率调度
            lr = lr_schedule[iteration]
            wd = wd_schedule[iteration]
            mom = momentum_schedule[iteration]
            teacher_temp = teacher_temp_schedule[iteration]
            last_layer_lr = last_layer_lr_schedule[iteration]
            
            # 更新优化器参数
            apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)
            
            if i % grad_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            
            # # 前向传播和反向传播
            # optimizer.zero_grad(set_to_none=True)
            
            # 计算开始时间
            data_time = time.time() - end
            
            # 执行前向-反向传播（只执行一次）
            loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

            # 规范化损失字典处理
            # if isinstance(loss_dict, tuple):
            #     loss_dict_processed = {}
            #     if len(loss_dict) > 0 and isinstance(loss_dict[0], dict):
            #         loss_dict_processed = loss_dict[0]
            #     else:
            #         for idx, loss_item in enumerate(loss_dict):
            #             if hasattr(loss_item, 'item'):
            #                 loss_dict_processed[f"loss_{idx}"] = loss_item
            #     loss_dict = loss_dict_processed
            if isinstance(loss_dict, tuple):
                    loss_dict = loss_dict[0] if isinstance(loss_dict[0], dict) else {"loss": loss_dict[0]}
    
            # # 计算损失总和
            # losses_reduced = 0
            # if isinstance(loss_dict, dict) and loss_dict:
            #     losses_reduced = sum(loss.item() if hasattr(loss, 'item') else float(loss) 
            #                         for loss in loss_dict.values())
            # elif hasattr(loss_dict, 'item'):
            #     losses_reduced = loss_dict.item()
            # else:
            #     try:
            #         losses_reduced = float(loss_dict)
            #     except (TypeError, ValueError):
            #         losses_reduced = 0
            #         logger.warning("无法计算损失总和，使用0替代")
            losses_reduced = 0
            if isinstance(loss_dict, dict) and loss_dict:
                # 缩放损失以适应梯度累积
                for k in loss_dict:
                    if hasattr(loss_dict[k], 'item'):
                        loss_dict[k] = loss_dict[k] / grad_accumulation_steps
                
                losses_reduced = sum(loss.item() if hasattr(loss, 'item') else float(loss) 
                                    for loss in loss_dict.values())
            
            # 只在完成累积或最后一个批次时更新权重
            if (i + 1) % grad_accumulation_steps == 0 or (i + 1) == len(dataloader):
                # 梯度裁剪
                if cfg.optim.clip_grad:
                    for key, module in model.student.items():
                        if args.distributed:
                            nn.utils.clip_grad_norm_(module.module.parameters(), cfg.optim.clip_grad)
                        else:
                            nn.utils.clip_grad_norm_(module.parameters(), cfg.optim.clip_grad)
                
                # 更新权重 - 移到条件内部
                if model.fp16_scaler is not None:
                    model.fp16_scaler.step(optimizer)
                    model.fp16_scaler.update()
                else:
                    optimizer.step()
            model.update_teacher(mom)
            # 更新教师模型
            optimizer.zero_grad(set_to_none=True)
            # 计算批次时间
            batch_time = time.time() - end
            end = time.time()

# 记录日志 - 仅在主进程
            if args.rank == 0 and (i % 20 == 0 or i == len(dataloader) - 1):
                log_str = f"Epoch: [{epoch+1}][{i}/{len(dataloader)}] " \
                        f"Time: {batch_time:.3f} ({data_time:.3f}) " \
                        f"Loss: {losses_reduced:.4f} " \
                        f"LR: {lr:.6f} WD: {wd:.6f} Mom: {mom:.6f}"
                
                # 添加损失详情
                if isinstance(loss_dict, dict):
                    for k, v in loss_dict.items():
                        try:
                            log_str += f" {k}: {v.item() if hasattr(v, 'item') else float(v):.4f}"
                        except (TypeError, ValueError):
                            log_str += f" {k}: NaN"
                
                # 日志输出
                logger.info(log_str)
            
            # 保存检查点 - 仅在主进程
            if args.rank == 0 and ((iteration + 1) % (cfg.train.saveckp_freq * len(dataloader)) == 0 or iteration + 1 == max_iter):
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    iteration=iteration,
                    cfg=cfg,
                    filename=f"checkpoint_epoch{epoch+1}_iter{iteration+1}.pth"
                )
                
                # 保存最新检查点
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    iteration=iteration,
                    cfg=cfg,
                    filename="checkpoint_latest.pth"
                )
            
            # # 保存单独的LoRA权重 - 仅在主进程
            # if args.rank == 0 and ((iteration + 1) % (cfg.evaluation.eval_period_iterations) == 0 or iteration + 1 == max_iter):
            #     lora_state_dict = {}
                
            #     # 提取所有LoRA相关的权重
            #     for name, param in model.named_parameters():
            #         if "lora_" in name and param.requires_grad:
            #             lora_state_dict[name] = param.data.clone()
                
            #     # 保存LoRA权重
            #     lora_path = os.path.join(cfg.train.output_dir, f"lora_weights_iter{iteration+1}.pt")
            #     torch.save(lora_state_dict, lora_path)
            #     logger.info(f"LoRA权重已保存到 {lora_path}")
    
    # 训练结束，保存最终模型 - 仅在主进程
    if args.rank == 0:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=cfg.optim.epochs - 1,
            iteration=max_iter - 1,
            cfg=cfg,
            filename="checkpoint_final.pth"
        )
        
        # 保存最终LoRA权重
        lora_state_dict = {}
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                lora_state_dict[name] = param.data.clone()
        
        # 保存LoRA权重
        lora_path = os.path.join(cfg.train.output_dir, "lora_weights_final.pt")
        torch.save(lora_state_dict, lora_path)
        logger.info(f"最终LoRA权重已保存到 {lora_path}")
    
    # 清理分布式环境
    if args.distributed:
        dist.destroy_process_group()
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()