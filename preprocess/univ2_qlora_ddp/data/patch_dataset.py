# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 您的组织名称
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
import logging
from typing import Callable, List, Optional, Tuple, Union
import random

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import numpy as np

logger = logging.getLogger("dinov2_qlora")

class PatchDataset(data.Dataset):
    """
    用于加载patch格式数据的数据集类
    
    目录结构：
    /remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/patches/
    ├── 1819360/
    │   ├── 146_315.png
    │   ├── 147_316.png
    │   └── ...
    ├── 另一个文件夹/
    │   └── ...
    └── ...
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        min_patches_per_class: int = 10,
        max_patches_per_class: int = 100,
        patch_size: int = 224,
    ):
        """
        初始化PatchDataset
        
        Args:
            root_dir: patch数据所在的根目录
            transform: 应用于图像的变换
            target_transform: 应用于目标的变换
            min_patches_per_class: 每个类别最少使用的patch数量
            max_patches_per_class: 每个类别最多使用的patch数量
            patch_size: patch的大小
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.min_patches_per_class = min_patches_per_class
        self.max_patches_per_class = max_patches_per_class
        self.patch_size = patch_size
        
        # 获取所有子文件夹（每个子文件夹代表一个类别）
        self.class_folders = [
            f for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f))
        ]
        
        # 收集所有图像路径
        self.image_paths = []
        self.class_indices = []
        
        for idx, class_folder in enumerate(self.class_folders):
            class_path = os.path.join(root_dir, class_folder)
            patch_files = [
                f for f in os.listdir(class_path)
                if f.endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(class_path, f))
            ]
            
            # 确保我们至少有min_patches_per_class个patch
            if len(patch_files) < self.min_patches_per_class:
                continue
                
            # 如果超过max_patches_per_class，随机采样
            if len(patch_files) > self.max_patches_per_class:
                patch_files = random.sample(patch_files, self.max_patches_per_class)
                
            for patch_file in patch_files:
                self.image_paths.append(os.path.join(class_path, patch_file))
                self.class_indices.append(idx)
        
        # 设置默认变换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.patch_size, self.patch_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        num_folders = len(self.class_folders)
        num_images = len(self.image_paths)
        num_unique_classes = len(set(self.class_indices))
        logger.info(f"创建了PatchDataset: 共有{num_folders}个图像文件夹, {num_unique_classes}个唯一类别, {num_images}个patch样本")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        获取一个数据样本
        
        Args:
            index: 索引
            
        Returns:
            图像张量和类别索引
        """
        img_path = self.image_paths[index]
        class_idx = self.class_indices[index]
        
        # 加载图像
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        except Exception as e:
            logger.error(f"加载图像出错 {img_path}: {e}")
            # 返回一个随机的其他索引
            return self.__getitem__(random.randint(0, len(self) - 1))
            
        # 应用变换
        if self.transform is not None:
            img = self.transform(img)
            
        target = class_idx
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target


def create_patch_dataset(
    root_dir: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    cache_file: Optional[str] = None,  # 新增参数
    use_cache: bool = True,            # 新增参数
    **kwargs
) -> PatchDataset:
    """
    创建一个PatchDataset实例
    
    Args:
        root_dir: patch数据所在的根目录
        transform: 应用于图像的变换
        target_transform: 应用于目标的变换
        cache_file: 缓存文件路径
        use_cache: 是否使用缓存
        kwargs: 其他参数传递给PatchDataset构造函数
        
    Returns:
        PatchDataset实例
    """
    # 检查是否使用缓存
    if use_cache and cache_file and os.path.exists(cache_file):
        logger.info(f"从缓存加载PatchDataset: {cache_file}")
        dataset = torch.load(cache_file)
        
        # 更新transform（因为缓存的数据集可能使用的是不同的transform）
        if transform is not None:
            dataset.transform = transform
            
        logger.info(f"从缓存加载成功: {len(dataset.image_paths)}个图像样本")
        return dataset
    
    # 否则创建新的数据集
    dataset = PatchDataset(
        root_dir=root_dir,
        transform=transform,
        target_transform=target_transform,
        **kwargs
    )
    
    # 保存缓存
    if use_cache and cache_file:
        logger.info(f"保存PatchDataset到缓存: {cache_file}")
        # 临时移除transform，因为它可能不可序列化
        temp_transform = dataset.transform
        dataset.transform = None
        # 确保目录存在
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(dataset, cache_file)
        # 恢复transform
        dataset.transform = temp_transform
    
    return dataset
