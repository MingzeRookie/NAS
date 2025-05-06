# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 您的组织名称
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter, ImageOps
import numpy as np


class GaussianBlur:
    """
    高斯模糊变换
    """
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class Solarization:
    """
    曝光效果变换
    """
    def __call__(self, x):
        return ImageOps.solarize(x)


class MultiCropTransform:
    """
    特别为patch数据设计的多尺度裁剪变换
    """
    def __init__(
        self,
        global_crops_scale: Tuple[float, float],
        local_crops_scale: Tuple[float, float],
        local_crops_number: int,
        global_crops_size: int = 224,
        local_crops_size: int = 96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        
        # 标准化参数
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        # 全局裁剪变换
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, interpolation=F.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, interpolation=F.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
            transforms.RandomApply([Solarization()], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        # 局部裁剪变换
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale, interpolation=F.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
    def __call__(self, image):
        """
        应用多尺度裁剪变换
        
        Args:
            image: 输入图像
            
        Returns:
            变换后的图像列表
        """
        all_crops = []
        
        # 应用第一个全局变换
        all_crops.append(self.global_transfo1(image))
        
        # 应用第二个全局变换
        all_crops.append(self.global_transfo2(image))
        
        # 应用局部变换
        for _ in range(self.local_crops_number):
            all_crops.append(self.local_transfo(image))
            
        return all_crops


def collate_multi_crop_batch(batch):
    """
    将多尺度裁剪批次整合为字典
    
    Args:
        batch: 输入批次，由MultiCropTransform生成
        
    Returns:
        包含全局和局部裁剪的字典
    """
    batch_size = len(batch)
    
    # 提取图像和标签
    all_images, all_labels = [], []
    for sample in batch:
        if isinstance(sample, tuple) and len(sample) == 2:
            images, label = sample
            all_images.append(images)
            all_labels.append(label)
        else:
            # 如果样本不是(图像, 标签)对，仅保留图像
            all_images.append(sample)
            all_labels.append(-1)  # 使用-1表示无标签
    
    # 检查第一个样本以确定全局和局部裁剪的数量
    n_global_crops = 2  # 通常是2个全局裁剪
    
    # 收集全局和局部裁剪
    global_crops, local_crops = [], []
    
    for images_tuple in all_images:
        # 添加全局裁剪
        for i in range(n_global_crops):
            global_crops.append(images_tuple[i])
        
        # 添加局部裁剪
        for i in range(n_global_crops, len(images_tuple)):
            local_crops.append(images_tuple[i])
    
    # 转换为张量
    if global_crops:
        global_crops = torch.stack(global_crops)
    else:
        global_crops = torch.tensor([])
        
    if local_crops:
        local_crops = torch.stack(local_crops)
    else:
        local_crops = torch.tensor([])
    
    # 组装结果字典
    result = {
        "collated_global_crops": global_crops,
        "collated_local_crops": local_crops,
        # 添加缺失的字段以兼容DINO-V2
        "collated_masks": torch.zeros(1),  # 占位符
        "mask_indices_list": torch.zeros(1, dtype=torch.long),  # 占位符
        "n_masked_patches": torch.zeros(1),  # 占位符
        "upperbound": 1,  # 占位符
        "masks_weight": torch.ones(1),  # 占位符
    }
    
    return result
