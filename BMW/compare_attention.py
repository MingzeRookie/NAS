# BMW/compare_attention.py

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import random
import logging
from PIL import Image
# --- 0. 环境与路径配置 ---

# 确保项目根目录在 sys.path 中
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
nas_main_path = os.path.abspath(os.path.join(current_path, '..'))
if nas_main_path not in sys.path:
    sys.path.insert(0, nas_main_path)

# 从您的项目中导入必要的模块
from BMW.src.models.attention_layers import SelfAttentionLayer, CrossAttentionLayer
from BMW.src.models.mil_aggregators import AttentionMIL

# --- 1. 用户配置区 ---
# 请根据您的实际情况修改以下路径
model_path_text = '/remote-home/share/lisj/Workspace/SOTA_NAS/BMW/outputs/2025-05-21-all/06-52-31/best_model.pth'
model_path_no_text = '/remote-home/share/lisj/Workspace/SOTA_NAS/BMW/outputs/2025-05-22/03-17-38/best_model.pth'

# b. 数据路径 (根据您最新提供的路径修改)
data_dir = '/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/MUSK-feature'
split_path = '/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/labels.csv'
text_features_path = '/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/MUSK-feature/MUSK-text-feature/averaged_text_feature.pt'


# c. 输出目录
output_dir = os.path.join(current_path, 'attention_comparison')

# d. 实验设置
PATCH_SIZE = 256
num_images_to_test = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. 关键代码重定义与修改 ---

# 2.1. 配置类
class SimpleConfigWindowParams:
    def __init__(self, **kwargs):
        self.patch_rows = kwargs.get('patch_rows', 3); self.patch_cols = kwargs.get('patch_cols', 3)
        self.stride_rows = kwargs.get('stride_rows', 2); self.stride_cols = kwargs.get('stride_cols', 2)
        self.num_selected_windows = kwargs.get('num_selected_windows', 20)
        self.pre_similarity_window_agg_type = kwargs.get('pre_similarity_window_agg_type', 'mean')
        self.light_agg_D = kwargs.get('light_agg_D', 128); self.light_agg_dropout = kwargs.get('light_agg_dropout', 0.25)

class SimpleConfigModelParams:
    def __init__(self, **kwargs):
        self.ablation_image_only = kwargs.get('ablation_image_only', False); self.ablation_no_window = kwargs.get('ablation_no_window', False)
        self.similarity_projection_dim = kwargs.get('similarity_projection_dim', 256); self.window_params = SimpleConfigWindowParams(**kwargs.get('window_params', {}))
        self.self_attn_heads = kwargs.get('self_attn_heads', 8); self.self_attn_dropout = kwargs.get('self_attn_dropout', 0.1)
        self.window_mil_output_dim = kwargs.get('window_mil_output_dim', 512); self.window_mil_D = kwargs.get('window_mil_D', 256) # 保持与 get_config 一致
        self.window_mil_dropout = kwargs.get('window_mil_dropout', 0.25); self.final_image_feature_dim = kwargs.get('final_image_feature_dim', 512)
        self.inter_window_mil_D = kwargs.get('inter_window_mil_D', 256); self.inter_window_mil_dropout = kwargs.get('inter_window_mil_dropout', 0.25) # 保持与 get_config 一致
        self.cross_attn_heads = kwargs.get('cross_attn_heads', 8); self.cross_attn_dropout = kwargs.get('cross_attn_dropout', 0.1)
        self.direct_cross_attention_embed_dim = kwargs.get('direct_cross_attention_embed_dim', 1024); self.direct_final_mil_output_dim = kwargs.get('direct_final_mil_output_dim', 512)
        self.direct_final_mil_hidden_dim = kwargs.get('direct_final_mil_hidden_dim', 256); self.direct_final_mil_dropout = kwargs.get('direct_final_mil_dropout', 0.25)
        self.classifier_hidden_dim = kwargs.get('classifier_hidden_dim', 512); self.classifier_dropout = kwargs.get('classifier_dropout', 0.25) # 保持与 get_config 一致
    def get(self, key, default=None): return getattr(self, key, default)

class SimpleConfig:
    def __init__(self, **kwargs):
        self.patch_feature_dim = kwargs.get('patch_feature_dim', 1024); self.text_feature_dim = kwargs.get('text_feature_dim', 768) # 与 get_config 中 text_feature_dim 匹配
        self.num_classes = kwargs.get('num_classes', 2); self.model_params = SimpleConfigModelParams(**kwargs.get('model_params', {})) # 与 get_config 中 num_classes 匹配

# 2.2. 重定义 MultimodalTextGuidedMIL 类 (与上一版相同)
class MultimodalTextGuidedMIL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config; self.patch_feature_dim = config.patch_feature_dim; self.num_classes = config.num_classes
        self.ablation_image_only = config.model_params.get('ablation_image_only', False); self.ablation_no_window = config.model_params.get('ablation_no_window', False)
        if self.ablation_image_only and self.ablation_no_window: raise ValueError("ablation_image_only and ablation_no_window cannot both be True.")
        if not self.ablation_image_only: self.text_feature_dim = config.text_feature_dim
        if not self.ablation_no_window:
            self.window_patch_rows = config.model_params.window_params.patch_rows; self.window_patch_cols = config.model_params.window_params.patch_cols
            self.patches_per_window = self.window_patch_rows * self.window_patch_cols; self.stride_rows = config.model_params.window_params.stride_rows
            self.stride_cols = config.model_params.window_params.stride_cols; self.num_selected_windows = config.model_params.window_params.num_selected_windows
            self.pre_similarity_window_agg_type = config.model_params.window_params.pre_similarity_window_agg_type
            if not self.ablation_image_only:
                self.similarity_projection_dim = config.model_params.similarity_projection_dim; self.text_proj_sim = nn.Linear(self.text_feature_dim, self.similarity_projection_dim)
                self.patch_proj_sim = nn.Linear(self.patch_feature_dim, self.similarity_projection_dim)
            if self.pre_similarity_window_agg_type == 'attention_light': self.light_window_aggregator = AttentionMIL(input_dim=self.patch_feature_dim, hidden_dim=config.model_params.window_params.light_agg_D, dropout_rate=config.model_params.window_params.light_agg_dropout, output_dim=self.patch_feature_dim)
            elif self.pre_similarity_window_agg_type == 'mean':
                def robust_mean_agg(x, mask):
                    if mask is None: return x.mean(dim=1)
                    float_mask = mask.unsqueeze(-1).float(); num_valid_patches = float_mask.sum(dim=1); masked_x = x * float_mask; sum_feats = masked_x.sum(dim=1); safe_num_valid_patches = num_valid_patches + 1e-8; aggregated_repr = sum_feats / safe_num_valid_patches
                    if torch.isnan(aggregated_repr).any(): aggregated_repr = torch.nan_to_num(aggregated_repr, nan=0.0, posinf=0.0, neginf=0.0)
                    return aggregated_repr
                self.light_window_aggregator = robust_mean_agg
            elif self.pre_similarity_window_agg_type == 'max': self.light_window_aggregator = lambda x, mask: x.masked_fill(~mask.unsqueeze(-1).bool(), -1e9).max(dim=1)[0] if mask is not None and mask.any() else x.max(dim=1)[0]
            elif self.ablation_image_only: pass
            else: raise ValueError(f"Unsupported pre_similarity_window_agg_type: {self.pre_similarity_window_agg_type}")
            self.window_self_attention = SelfAttentionLayer(embed_dim=self.patch_feature_dim, num_heads=config.model_params.self_attn_heads, dropout=config.model_params.self_attn_dropout); self.window_mil_output_dim = config.model_params.window_mil_output_dim
            self.window_mil_aggregator = AttentionMIL(input_dim=self.patch_feature_dim, hidden_dim=config.model_params.window_mil_D, dropout_rate=config.model_params.window_mil_dropout, output_dim=self.window_mil_output_dim); self.final_image_feature_dim = config.model_params.final_image_feature_dim
            self.inter_window_aggregator = AttentionMIL(input_dim=self.window_mil_output_dim, hidden_dim=config.model_params.inter_window_mil_D, dropout_rate=config.model_params.inter_window_mil_dropout, output_dim=self.final_image_feature_dim)
        if not self.ablation_image_only:
            cross_attn_query_dim = self.final_image_feature_dim if not self.ablation_no_window else self.patch_feature_dim; cross_attention_embed_dim = config.model_params.get('direct_cross_attention_embed_dim', self.patch_feature_dim if self.ablation_no_window else self.final_image_feature_dim)
            self.cross_attention = CrossAttentionLayer(query_dim=cross_attn_query_dim, key_dim=self.text_feature_dim, embed_dim=cross_attention_embed_dim, num_heads=config.model_params.cross_attn_heads, dropout=config.model_params.cross_attn_dropout)
            if self.ablation_no_window: final_mil_output_dim = config.model_params.get('direct_final_mil_output_dim', 512); self.final_mil_aggregator = AttentionMIL(input_dim=cross_attention_embed_dim, hidden_dim=config.model_params.get('direct_final_mil_hidden_dim', 128), dropout_rate=config.model_params.get('direct_final_mil_dropout', 0.25), output_dim=final_mil_output_dim); self.fused_feature_dim = final_mil_output_dim
            else: self.fused_feature_dim = self.final_image_feature_dim + cross_attention_embed_dim
        else: self.cross_attention = None; self.fused_feature_dim = self.final_image_feature_dim
        self.classifier = nn.Sequential(nn.Linear(self.fused_feature_dim, config.model_params.classifier_hidden_dim), nn.ReLU(), nn.Dropout(config.model_params.classifier_dropout), nn.Linear(config.model_params.classifier_hidden_dim, self.num_classes))

# 在 MultimodalTextGuidedMIL 类中
    def _generate_candidate_spatial_windows(self, all_patch_features_wsi, all_patch_grid_indices_wsi, wsi_grid_shape, batch_item_idx_for_debug=None):
        if self.ablation_no_window: return [], [], [], [] # 返回四个空列表
        if all_patch_features_wsi is None or all_patch_features_wsi.numel() == 0: return [], [], [], []
        
        max_r, max_c = wsi_grid_shape[0].item(), wsi_grid_shape[1].item()
        candidate_windows_feats_list = []
        candidate_windows_masks_list = []
        candidate_windows_top_left_coords_list = [] # 存储窗口左上角网格坐标
        candidate_windows_patch_indices_list = []   # ***【新增】*** 存储每个窗口包含的原始 patch 索引

        if all_patch_grid_indices_wsi.numel() == 0: return [], [], [], []
        try:
            # 创建一个从 (行,列) 网格索引到原始 patch 列表索引的映射
            coord_to_idx_map = {tuple(coord.tolist()): i for i, coord in enumerate(all_patch_grid_indices_wsi)}
        except Exception: # 更通用的异常捕获
            return [], [], [], []
            
        eff_window_rows = min(self.window_patch_rows, max_r)
        eff_window_cols = min(self.window_patch_cols, max_c)

        if eff_window_rows <= 0 or eff_window_cols <= 0: return [], [], [], []

        for r_start in range(0, max_r - eff_window_rows + 1, self.stride_rows):
            for c_start in range(0, max_c - eff_window_cols + 1, self.stride_cols):
                current_window_original_patch_indices = [] # 存储此窗口内的原始 patch 索引
                
                for r_offset in range(eff_window_rows): 
                    for c_offset in range(eff_window_cols):
                        abs_r, abs_c = r_start + r_offset, c_start + c_offset
                        if (abs_r, abs_c) in coord_to_idx_map:
                            current_window_original_patch_indices.append(coord_to_idx_map[(abs_r, abs_c)])
                
                if len(current_window_original_patch_indices) > 0:
                    window_feats = all_patch_features_wsi[current_window_original_patch_indices]
                    num_actual_patches = window_feats.shape[0]
                    
                    padded_window_feats = torch.zeros(self.patches_per_window, self.patch_feature_dim,
                                                      device=all_patch_features_wsi.device, dtype=all_patch_features_wsi.dtype)
                    window_mask = torch.zeros(self.patches_per_window, device=all_patch_features_wsi.device, dtype=torch.bool)
                    
                    num_to_fill = min(num_actual_patches, self.patches_per_window)
                    if num_to_fill > 0: 
                        padded_window_feats[:num_to_fill] = window_feats[:num_to_fill] # 使用实际提取到的 patch 特征
                        window_mask[:num_to_fill] = True
                        
                    candidate_windows_feats_list.append(padded_window_feats)
                    candidate_windows_masks_list.append(window_mask)
                    candidate_windows_top_left_coords_list.append(torch.tensor([r_start, c_start], device=all_patch_features_wsi.device))
                    candidate_windows_patch_indices_list.append(torch.tensor(current_window_original_patch_indices, device=all_patch_features_wsi.device, dtype=torch.long)) # ***【新增】***
        
        return candidate_windows_feats_list, candidate_windows_masks_list, candidate_windows_top_left_coords_list, candidate_windows_patch_indices_list

# 在 MultimodalTextGuidedMIL 类中
    def _select_windows(self, all_patch_features_wsi, text_feat_wsi, all_patch_grid_indices_wsi, wsi_grid_shape, batch_item_idx_for_debug=None, return_debug_info=False):
        debug_info = {} 
        def return_zeros_for_select(dev_ref, dtype_ref):
            # ... (此辅助函数不变)
            dev = dev_ref.device if dev_ref is not None else torch.device('cpu'); dtype = dtype_ref.dtype if dtype_ref is not None else torch.float32
            feats = torch.zeros(self.num_selected_windows, self.patches_per_window, self.patch_feature_dim, device=dev, dtype=dtype)
            masks = torch.zeros(self.num_selected_windows, self.patches_per_window, device=dev, dtype=torch.bool)
            return feats, masks

        if all_patch_features_wsi is None or all_patch_features_wsi.numel() == 0: return return_zeros_for_select(all_patch_features_wsi, all_patch_features_wsi), debug_info
        
        # ***【修改】*** 接收新的返回
        candidate_windows_feats_list, candidate_windows_masks_list, \
        candidate_windows_top_left_coords_list, candidate_windows_patch_indices_list = \
            self._generate_candidate_spatial_windows(all_patch_features_wsi, all_patch_grid_indices_wsi, wsi_grid_shape, batch_item_idx_for_debug)

        if not candidate_windows_feats_list: return return_zeros_for_select(all_patch_features_wsi, all_patch_features_wsi), debug_info
        
        final_selected_feats_list, final_selected_masks_list = [], []
        # ***【新增】*** 存储被选中窗口的原始 patch 索引和窗口左上角坐标
        final_selected_patch_indices_list = []
        final_selected_window_top_left_coords_list = []


        num_candidates, num_to_select = len(candidate_windows_feats_list), min(self.num_selected_windows, len(candidate_windows_feats_list))

        if self.ablation_image_only or text_feat_wsi is None: # 随机选择
            if num_to_select > 0:
                selected_indices = random.sample(range(num_candidates), num_to_select) if num_candidates > num_to_select else list(range(num_candidates))
                for idx in selected_indices: 
                    final_selected_feats_list.append(candidate_windows_feats_list[idx])
                    final_selected_masks_list.append(candidate_windows_masks_list[idx])
                    final_selected_window_top_left_coords_list.append(candidate_windows_top_left_coords_list[idx]) # ***【新增】***
                    final_selected_patch_indices_list.append(candidate_windows_patch_indices_list[idx]) # ***【新增】***
                if return_debug_info: 
                    if final_selected_window_top_left_coords_list:
                        debug_info['selected_window_top_left_coords'] = torch.stack(final_selected_window_top_left_coords_list)
                    debug_info['selected_windows_patch_indices'] = final_selected_patch_indices_list # 列表的列表
        else: # 文本引导选择
            aggregated_candidate_reprs, valid_candidate_indices = [], []
            for i, window_feats in enumerate(candidate_windows_feats_list):
                # ... (聚合逻辑不变)
                if not candidate_windows_masks_list[i].any(): continue
                if self.pre_similarity_window_agg_type == 'attention_light': agg_repr = self.light_window_aggregator(window_feats.unsqueeze(0), instance_mask=candidate_windows_masks_list[i].unsqueeze(0))[0]
                else: agg_repr = self.light_window_aggregator(window_feats.unsqueeze(0), candidate_windows_masks_list[i].unsqueeze(0))
                aggregated_candidate_reprs.append(agg_repr.squeeze(0)); valid_candidate_indices.append(i)

            if not aggregated_candidate_reprs: return return_zeros_for_select(all_patch_features_wsi, all_patch_features_wsi), debug_info
            
            stacked_candidate_reprs = torch.stack(aggregated_candidate_reprs)
            proj_text_feat = self.text_proj_sim(text_feat_wsi)
            proj_candidate_reprs = self.patch_proj_sim(stacked_candidate_reprs)
            similarity_scores = F.cosine_similarity(proj_text_feat.unsqueeze(0), proj_candidate_reprs, dim=1)

            if return_debug_info: 
                if valid_candidate_indices:
                    debug_info['candidate_window_top_left_coords'] = torch.stack([candidate_windows_top_left_coords_list[i] for i in valid_candidate_indices])
                    debug_info['candidate_windows_patch_indices'] = [candidate_windows_patch_indices_list[i] for i in valid_candidate_indices] # 列表的列表
                debug_info['similarity_scores'] = similarity_scores
            
            num_to_select_topk = min(self.num_selected_windows, similarity_scores.shape[0])
            if num_to_select_topk > 0:
                _, top_k_relative_indices = torch.topk(similarity_scores, k=num_to_select_topk, dim=0)
                for original_idx_in_valid_candidates in top_k_relative_indices.tolist():
                    absolute_idx_in_all_candidates = valid_candidate_indices[original_idx_in_valid_candidates]
                    final_selected_feats_list.append(candidate_windows_feats_list[absolute_idx_in_all_candidates])
                    final_selected_masks_list.append(candidate_windows_masks_list[absolute_idx_in_all_candidates])
                    # 注意：如果文本引导也需要记录被选中的窗口的 patch 索引和坐标，可以在这里添加
                    # final_selected_patch_indices_list.append(candidate_windows_patch_indices_list[absolute_idx_in_all_candidates])
                    # final_selected_window_top_left_coords_list.append(candidate_windows_top_left_coords_list[absolute_idx_in_all_candidates])


        # Padding (不变)
        # ...
        padded_selected_feats = torch.stack(final_selected_feats_list, dim=0) if final_selected_feats_list else torch.zeros(0, self.patches_per_window, self.patch_feature_dim, device=all_patch_features_wsi.device, dtype=all_patch_features_wsi.dtype)
        padded_selected_masks = torch.stack(final_selected_masks_list, dim=0) if final_selected_masks_list else torch.zeros(0, self.patches_per_window, device=all_patch_features_wsi.device, dtype=torch.bool)
        if (num_padding := self.num_selected_windows - padded_selected_feats.shape[0]) > 0:
            padding_f = torch.zeros(num_padding, self.patches_per_window, self.patch_feature_dim, device=padded_selected_feats.device, dtype=padded_selected_feats.dtype)
            padding_m = torch.zeros(num_padding, self.patches_per_window, device=padded_selected_feats.device, dtype=torch.bool)
            padded_selected_feats = torch.cat([padded_selected_feats, padding_f], dim=0); padded_selected_masks = torch.cat([padded_selected_masks, padding_m], dim=0)
        
        return padded_selected_feats, padded_selected_masks, debug_info

    def forward(self, image_patch_features_batch, patch_grid_indices_batch, grid_shapes_batch, text_feat_batch=None, original_patch_coordinates_batch=None, patch_mask_batch=None, return_debug_info=False):
        batch_size = image_patch_features_batch.shape[0]; all_batch_debug_info = []
        if self.ablation_no_window:
            pass 
        else:
            all_selected_windows_feats_b, all_selected_windows_masks_b = [], []
            for i in range(batch_size):
                current_patch_feats, current_grid_indices = image_patch_features_batch[i], patch_grid_indices_batch[i]
                current_text_feat_sample = text_feat_batch[i] if text_feat_batch is not None and not self.ablation_image_only else None
                if current_patch_feats.shape[0] == 0:
                    s_feats = torch.zeros(self.num_selected_windows, self.patches_per_window, self.patch_feature_dim, device=current_patch_feats.device, dtype=current_patch_feats.dtype)
                    s_mask = torch.zeros(self.num_selected_windows, self.patches_per_window, device=current_patch_feats.device, dtype=torch.bool); debug_info = {}
                else: s_feats, s_mask, debug_info = self._select_windows(current_patch_feats, current_text_feat_sample, current_grid_indices, grid_shapes_batch[i], batch_item_idx_for_debug=i, return_debug_info=return_debug_info)
                all_selected_windows_feats_b.append(s_feats); all_selected_windows_masks_b.append(s_mask); all_batch_debug_info.append(debug_info)
            selected_windows_feats = torch.stack(all_selected_windows_feats_b, dim=0); selected_windows_mask = torch.stack(all_selected_windows_masks_b, dim=0)
            k_w = self.num_selected_windows; proc_windows_feats = selected_windows_feats.view(batch_size * k_w, self.patches_per_window, self.patch_feature_dim); proc_windows_mask = selected_windows_mask.view(batch_size * k_w, self.patches_per_window)
            current_key_padding_mask = ~proc_windows_mask.bool() if proc_windows_mask is not None else None; attended_patch_feats = torch.zeros_like(proc_windows_feats)
            not_fully_masked_window_indices = (~current_key_padding_mask.all(dim=1)).nonzero(as_tuple=True)[0] if current_key_padding_mask is not None else torch.arange(proc_windows_feats.shape[0], device=proc_windows_feats.device)
            if not_fully_masked_window_indices.numel() > 0:
                output_from_attention = self.window_self_attention(proc_windows_feats[not_fully_masked_window_indices], key_padding_mask=current_key_padding_mask[not_fully_masked_window_indices] if current_key_padding_mask is not None else None)
                attended_patch_feats.index_copy_(0, not_fully_masked_window_indices, output_from_attention)
            aggregated_window_reprs, _ = self.window_mil_aggregator(attended_patch_feats, instance_mask=proc_windows_mask)
            aggregated_window_reprs = aggregated_window_reprs.view(batch_size, k_w, self.window_mil_output_dim)
            final_image_repr, _ = self.inter_window_aggregator(aggregated_window_reprs, instance_mask=selected_windows_mask.any(dim=2))
            if not self.ablation_image_only and self.cross_attention is not None:
                current_text_key_value = text_feat_batch.unsqueeze(1) # 形状变为 [batch_size, 1, text_feature_dim]

                fused_representation_valid = self.cross_attention(query=final_image_repr.unsqueeze(1), key_value=current_text_key_value)
                final_batch_representation = torch.cat([final_image_repr, fused_representation_valid.squeeze(1)], dim=-1)
            else: final_batch_representation = final_image_repr
        logits = self.classifier(final_batch_representation)
        if return_debug_info: return logits, all_batch_debug_info
        return logits

# --- 3. 适配后的数据加载器 ---
class WsiDatasetForAttention(Dataset):
    def __init__(self, data_dir, labels_df, text_features_path, 
                 patch_source_base_dir, # 改为 patch 图像的顶级父目录，例如 /remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/patches/
                 patch_size_on_wsi, # 这是您之前定义的 PATCH_SIZE，指 WSI 上的大小
                 n_patches_to_sample=4096, 
                 text_feature_dim=1024,
                 # 新增参数，用于指定是在 train, test 还是其他子目录下寻找 patch 图像
                 patch_subdir='train' # 默认为 'train'
                 ):
        self.data_dir = data_dir # 用于加载 .pt 特征文件
        self.labels_df = labels_df
        self.text_features_path = text_features_path
        self.patch_source_base_dir = patch_source_base_dir # 例如 /remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/patches/
        self.patch_size_on_wsi = patch_size_on_wsi # WSI 上的 patch 尺寸，用于显示
        self.n_patches_to_sample = n_patches_to_sample
        self.text_feature_dim = text_feature_dim
        self.patch_subdir = patch_subdir # 例如 'train' 或 'test'

        if os.path.exists(self.text_features_path):
            self.averaged_text_feature = torch.load(self.text_features_path, map_location='cpu')
            if not isinstance(self.averaged_text_feature, torch.Tensor):
                raise TypeError(f"Expected text_features_path to contain a single tensor, but got {type(self.averaged_text_feature)}")
            self.averaged_text_feature = self.averaged_text_feature.float().squeeze()
            if self.averaged_text_feature.ndim == 0 or self.averaged_text_feature.shape[0] != self.text_feature_dim:
                raise ValueError(f"Loaded averaged text feature has shape {self.averaged_text_feature.shape}, "
                                 f"but expected a 1D tensor of size {self.text_feature_dim}.")
            print(f"Successfully loaded averaged text feature from {self.text_features_path} with shape {self.averaged_text_feature.shape}")
        else:
            print(f"Warning: Averaged text features file not found at {self.text_features_path}. Using zero vector.")
            self.averaged_text_feature = torch.zeros(self.text_feature_dim)

        if 'ID' in self.labels_df.columns:
            self.labels_df['ID'] = self.labels_df['ID'].astype(str)
        else:
            raise ValueError("CSV file must contain an 'ID' column.")

    def __len__(self): return len(self.labels_df)

    def __getitem__(self, index):
        row = self.labels_df.loc[index]
        slide_id = str(row['ID'])
        text_embeds_value = self.averaged_text_feature

        wsi_feature_file_path = os.path.join(self.data_dir, f"{slide_id}.pt") # 这是特征.pt文件，不是图像patch的.pt
        try:
            wsi_data = torch.load(wsi_feature_file_path, map_location='cpu')
        except FileNotFoundError:
            print(f"Error: Cannot find feature .pt file for slide_id {slide_id} at {wsi_feature_file_path}")
            return {'slide_id': slide_id, 'wsi_feats': torch.empty(0, self.text_feature_dim), 
                    'coords': torch.empty(0, 2), 'patch_images': np.array([]), 
                    'grid_indices': torch.empty(0,2), 'grid_shape': torch.tensor([0,0]), 
                    'text_embeds': text_embeds_value, 'label': 'N/A'}

        wsi_feats = wsi_data['bag_feats'].float()
        coords_all = wsi_data['coords'] # 这是该 slide 所有 patch 的坐标列表
        
        num_original_patches = len(wsi_feats)
        
        # 根据原始坐标加载所有 patch 图像
        loaded_patch_images_all = []
        if self.patch_source_base_dir:
            slide_patch_dir = os.path.join(self.patch_source_base_dir, self.patch_subdir, slide_id)
            if not os.path.isdir(slide_patch_dir):
                print(f"Warning: Patch directory for slide {slide_id} not found at {slide_patch_dir}. Cannot load patch images.")
                patch_images_all = np.array([]) # 标记为无法加载
            else:
                for i in range(num_original_patches):
                    # 从 coords_all 获取当前 patch 的坐标
                    y_coord, x_coord = coords_all[i, 0].item(), coords_all[i, 1].item()
                    patch_image_filename = f"{y_coord}_{x_coord}.png"
                    patch_image_path = os.path.join(slide_patch_dir, patch_image_filename)
                    
                    try:
                        img = Image.open(patch_image_path).convert('RGB')
                        loaded_patch_images_all.append(np.array(img))
                    except FileNotFoundError:
                        # print(f"Warning: Patch image not found: {patch_image_path}. Using placeholder.")
                        # 对于可视化，如果patch图像缺失，最好还是用一个占位符，比如纯色块
                        # 这里我们先简单地用一个黑色占位符，尺寸需要和真实patch图像一致（如果知道的话）
                        # 或者让可视化函数处理这种情况
                        # 为了简单，我们假设patch图像的显示尺寸会由patch_size_on_wsi决定，所以这里先不创建占位图像，让列表长度不一致
                        # 一个更好的方法是确保所有patch都有图像，或者可视化时能处理缺失
                        pass # 跳过缺失的图像，会导致 loaded_patch_images_all 长度不一

                if len(loaded_patch_images_all) == num_original_patches:
                    patch_images_all = np.stack(loaded_patch_images_all)
                elif len(loaded_patch_images_all) > 0: # 如果部分加载成功，只用加载成功的
                     print(f"Warning: Loaded {len(loaded_patch_images_all)} images for {num_original_patches} patches for slide {slide_id}. Some patch images might be missing.")
                     # 这种情况比较复杂，因为特征和坐标是全的，但图像是部分的。
                     # 最简单粗暴的方式是，如果数量不匹配，就不使用图像叠加。
                     patch_images_all = np.array([]) # 标记为不使用图像叠加
                else:
                    patch_images_all = np.array([])
        else:
            patch_images_all = np.array([])

        # 进行采样 (如果需要)
        sampled_wsi_feats = wsi_feats
        sampled_coords = coords_all
        sampled_patch_images = patch_images_all

        if self.n_patches_to_sample > 0 and num_original_patches > self.n_patches_to_sample:
            indices = np.random.choice(num_original_patches, self.n_patches_to_sample, replace=False)
            sampled_wsi_feats = wsi_feats[indices]
            sampled_coords = coords_all[indices]
            if patch_images_all.ndim > 1 and len(patch_images_all) == num_original_patches:
                sampled_patch_images = patch_images_all[indices]
            elif len(patch_images_all) != len(sampled_wsi_feats) : # 如果图像数量在采样后不匹配，则不使用
                 sampled_patch_images = np.array([])


        # 后续处理使用采样后的数据
        grid_indices = torch.empty((0,2), dtype=torch.long)
        grid_shape = torch.tensor([0,0], dtype=torch.long)
        if sampled_coords.numel() > 0 :
            # 注意：这里的 patch_size_on_wsi 是用于计算网格索引的，可能与patch图像的实际像素尺寸不同
            grid_indices = (sampled_coords / self.patch_size_on_wsi).round().long() 
            if grid_indices.numel() > 0: grid_shape = grid_indices.max(dim=0)[0] + 1
            else: grid_shape = torch.tensor([1,1], dtype=torch.long) 
        else: grid_shape = torch.tensor([1,1], dtype=torch.long) 
            
        inflammation_label = str(row.get('inflammation', 'N/A'))

        return {'slide_id': slide_id, 'wsi_feats': sampled_wsi_feats, 'coords': sampled_coords, 
                'patch_images': sampled_patch_images, 
                'grid_indices': grid_indices, 'grid_shape': grid_shape, 
                'text_embeds': text_embeds_value, 'label': inflammation_label}

# --- 4. 可视化函数 (不叠加，只画注意力色块) ---# --- 4. 可视化函数 (优化版 - 只画注意力色块，尝试消除“框”感) ---# --- 4. 可视化函数 (优化版 - 只画注意力色块，尝试消除“框”感) ---
def visualize_patch_attention(ax, title, all_patch_coords_cpu, patch_scores_cpu, patch_images_np, patch_size_const):
    # patch_images_np 参数保留以保持函数签名一致性，但在此版本中不使用它
    if all_patch_coords_cpu.numel() == 0:
        ax.set_title(title + " (No patches)")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    norm_scores = patch_scores_cpu.clone().cpu() # 确保在CPU上
    if norm_scores.numel() > 0:
        min_score, max_score = norm_scores.min(), norm_scores.max()
        if max_score > min_score:
            norm_scores = (norm_scores - min_score) / (max_score - min_score)
        elif max_score > 0 : # 如果所有值都一样且大于0，则都设为1
            norm_scores[:] = 1.0
        else: # 如果所有值都一样且为0 (或负数)，则都设为0
            norm_scores[:] = 0.0
    else: # 如果 patch_scores_cpu 为空
        norm_scores = torch.tensor([])


    cmap = plt.cm.jet # 使用 jet 色谱

    # 确定边界，以便设置正确的绘图范围
    if all_patch_coords_cpu.numel() > 0:
        # 假设 coords 是 [y, x] (row, col)
        min_x_coord = all_patch_coords_cpu[:, 1].min().item()
        max_x_coord = all_patch_coords_cpu[:, 1].max().item() + patch_size_const
        min_y_coord = all_patch_coords_cpu[:, 0].min().item()
        max_y_coord = all_patch_coords_cpu[:, 0].max().item() + patch_size_const
        # 稍微扩大一点边界，确保所有 patch 完整显示
        ax.set_xlim(min_x_coord - patch_size_const * 0.05, max_x_coord + patch_size_const * 0.05) 
        ax.set_ylim(max_y_coord + patch_size_const * 0.05, min_y_coord - patch_size_const * 0.05)
    else: # 如果没有 patch 坐标，设置一个默认视图
        ax.set_xlim(0, patch_size_const * 10) 
        ax.set_ylim(patch_size_const * 10, 0)
    
    # 绘制代表每个 patch 的、根据注意力分数着色的矩形
    for i in range(all_patch_coords_cpu.shape[0]):
        coords = all_patch_coords_cpu[i]
        # 安全地获取分数，如果 norm_scores 为空或索引越界，则使用0
        score = norm_scores[i].item() if i < len(norm_scores) and norm_scores.numel() > 0 else 0.0
        
        x_coord = coords[1].item() # Column index for x-coordinate
        y_coord = coords[0].item() # Row index for y-coordinate

        attention_color = cmap(score) # 获取颜色 (R, G, B, A_cmap)
        
        rect = plt.Rectangle(
            (x_coord, y_coord),             # 矩形左下角 (x,y) - Matplotlib 默认 (x,y) 是左下角
            patch_size_const,               # 宽度
            patch_size_const,               # 高度
            linewidth=0,                    # 设置为0以避免 patch 间的缝隙
            alpha=1.0,                      # 完全不透明
            facecolor=attention_color[:3]   # 只取 RGB 颜色值
        )
        ax.add_patch(rect)

    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis() # Y 轴向下，符合图像和数组索引的习惯
    # 移除坐标轴刻度和标签，使图像更干净
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off') # 完全关闭坐标轴框架

# --- 5. 主逻辑 ---
def get_config(
    patch_feature_dim=1024, text_feature_dim=768, num_classes=2, # 确保这些基础维度正确
    ablation_image_only=False, similarity_projection_dim=256,
    wp_patch_rows=3, wp_patch_cols=3, wp_stride_rows=2, wp_stride_cols=2,
    wp_num_selected_windows=20, wp_pre_similarity_window_agg_type='mean',
    wp_light_agg_D=128, self_attn_heads=8, window_mil_output_dim=512,
    window_mil_D=256, # 根据您的 MultimodalTextGuidedMIL 类定义调整
    final_image_feature_dim=512, inter_window_mil_D=256, # 根据您的 MultimodalTextGuidedMIL 类定义调整
    cross_attn_heads=8, direct_cross_attention_embed_dim=1024, # 假设值
    classifier_hidden_dim=512, # 根据您的 MultimodalTextGuidedMIL 类定义调整
    **other_model_params 
    ):
    return SimpleConfig(
        patch_feature_dim=patch_feature_dim, text_feature_dim=text_feature_dim, num_classes=num_classes,
        model_params={
            'ablation_image_only': ablation_image_only, 'ablation_no_window': False, 
            'similarity_projection_dim': similarity_projection_dim,
            'window_params': {
                'patch_rows': wp_patch_rows, 'patch_cols': wp_patch_cols,
                'stride_rows': wp_stride_rows, 'stride_cols': wp_stride_cols,
                'num_selected_windows': wp_num_selected_windows,
                'pre_similarity_window_agg_type': wp_pre_similarity_window_agg_type,
                'light_agg_D': wp_light_agg_D, 'light_agg_dropout': 0.25,
            },
            'self_attn_heads': self_attn_heads, 'self_attn_dropout': 0.1,
            'window_mil_output_dim': window_mil_output_dim, 'window_mil_D': window_mil_D,
            'window_mil_dropout': 0.25, 'final_image_feature_dim': final_image_feature_dim,
            'inter_window_mil_D': inter_window_mil_D, 'inter_window_mil_dropout': 0.25,
            'cross_attn_heads': cross_attn_heads, 'cross_attn_dropout': 0.1,
            'direct_cross_attention_embed_dim': direct_cross_attention_embed_dim,
            'classifier_hidden_dim': classifier_hidden_dim, 'classifier_dropout': 0.25,
            **other_model_params
        }
    )

def main():
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading models...")
    
    # 您需要根据训练两个模型时的实际配置来调整这些参数
    # 特别注意报错中提示维度不匹配的参数
    config_params_text = {
        "patch_feature_dim": 1024, 
        "text_feature_dim": 1024,  
        "num_classes": 4,          
        "ablation_image_only": False,
        "similarity_projection_dim": 256, 
        "wp_light_agg_D": 128, 
        "window_mil_D": 128,   
        "inter_window_mil_D": 128, 
        "direct_cross_attention_embed_dim": 512, 
        "classifier_hidden_dim": 256, 
        "wp_patch_rows":3, "wp_patch_cols":3, "wp_stride_rows":2, "wp_stride_cols":2,
        "wp_num_selected_windows":20, "wp_pre_similarity_window_agg_type":'mean',
        "self_attn_heads":8, "window_mil_output_dim":512,
        "final_image_feature_dim":512, "cross_attn_heads":8,
    }
    config_text = get_config(**config_params_text)
    model_text = MultimodalTextGuidedMIL(config_text)
    checkpoint_text = torch.load(model_path_text, map_location='cpu')
    model_text.load_state_dict(checkpoint_text['model_state_dict'])
    model_text.to(device).eval()

    config_params_no_text = config_params_text.copy() 
    config_params_no_text["ablation_image_only"] = True
    
    config_no_text = get_config(**config_params_no_text)
    model_no_text = MultimodalTextGuidedMIL(config_no_text)
    checkpoint_no_text = torch.load(model_path_no_text, map_location='cpu')
    model_no_text.load_state_dict(checkpoint_no_text['model_state_dict'])
    model_no_text.to(device).eval()
    print("Models loaded.")
    patch_images_source_directory = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/patches/" 
    current_patch_subdir = 'train' 

    text_feat_dim_for_dataset = config_params_text.get("text_feature_dim", 1024)
    all_labels_df = pd.read_csv(split_path)
    dataset = WsiDatasetForAttention(
        data_dir=data_dir, # 这是 .pt 特征文件的目录
        labels_df=all_labels_df,
        text_features_path=text_features_path,
        patch_source_base_dir=patch_images_source_directory, # <--- 新参数
        patch_size_on_wsi=PATCH_SIZE, # 这是您之前定义的 PATCH_SIZE
        n_patches_to_sample=num_images_to_test if num_images_to_test < 500 else 4096, # 如果测试图片少，就取测试图片数，否则取默认值
        text_feature_dim=text_feat_dim_for_dataset,
        patch_subdir=current_patch_subdir # <--- 新参数
    )
    print(f"Dataset loaded. Total images from CSV: {len(dataset)}. Will process first {num_images_to_test} images.")

    for i in tqdm(range(min(num_images_to_test, len(dataset)))):
        data = dataset[i]; slide_id = data['slide_id']
        feats = data['wsi_feats'].unsqueeze(0).to(device)
        grid_indices = data['grid_indices'].unsqueeze(0).to(device)
        grid_shape = data['grid_shape'].unsqueeze(0).to(device)
        text_embeds = data['text_embeds'].unsqueeze(0).to(device) # 确保 text_embeds 有 batch 维度
        
# 在 main 函数的循环中
# ... (获取 data, slide_id, feats, grid_indices, grid_shape, text_embeds 之后)
        with torch.no_grad():
            _, debug_info_list_text = model_text(feats, grid_indices, grid_shape, text_feat_batch=text_embeds, return_debug_info=True)
            _, debug_info_list_no_text = model_no_text(feats, grid_indices, grid_shape, text_feat_batch=text_embeds, return_debug_info=True)
        
        debug_info_text = debug_info_list_text[0] # batch size is 1
        debug_info_no_text = debug_info_list_no_text[0]

        num_all_patches = data['wsi_feats'].shape[0]
        patch_scores_text = torch.zeros(num_all_patches, device='cpu')
        patch_scores_no_text = torch.zeros(num_all_patches, device='cpu')

        # 处理文本引导模型的 patch 分数
        if 'similarity_scores' in debug_info_text and 'candidate_windows_patch_indices' in debug_info_text:
            candidate_patch_indices_list = debug_info_text['candidate_windows_patch_indices'] # list of tensors
            similarity_scores = debug_info_text['similarity_scores'].cpu() # tensor
            
            for i_window, patch_indices_in_window_tensor in enumerate(candidate_patch_indices_list):
                if i_window < len(similarity_scores): # 确保索引有效
                    score_for_this_window = similarity_scores[i_window]
                    patch_indices_in_window = patch_indices_in_window_tensor.cpu().tolist()
                    for patch_idx in patch_indices_in_window:
                        if patch_idx < num_all_patches:
                             # 如果一个 patch 属于多个重叠窗口，可以选择最大/平均得分，这里简单覆盖
                            patch_scores_text[patch_idx] = max(patch_scores_text[patch_idx], score_for_this_window)


        # 处理无文本模型的 patch 分数 (高亮被选中的窗口内的patches)
        if 'selected_windows_patch_indices' in debug_info_no_text:
            selected_patch_indices_list = debug_info_no_text['selected_windows_patch_indices'] # list of tensors
            for patch_indices_in_window_tensor in selected_patch_indices_list:
                patch_indices_in_window = patch_indices_in_window_tensor.cpu().tolist()
                for patch_idx in patch_indices_in_window:
                    if patch_idx < num_all_patches:
                        patch_scores_no_text[patch_idx] = 1.0 # 赋予一个高分

        # d. 可视化 (传递 patch_scores 给 visualize_attention)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f'Patch-Level Attention for Slide ID: {slide_id}', fontsize=16)
        
        # visualize_attention 函数需要修改以接收 patch_scores
        patch_images_for_plot = data.get('patch_images', np.array([])) # 安全获取
        visualize_patch_attention(ax1, 'Text-Guided Attention', data['coords'].cpu(), patch_scores_text, PATCH_SIZE)
        visualize_patch_attention(ax2, 'Image-Only (Selected Windows Patches)', data['coords'].cpu(), patch_scores_no_text, PATCH_SIZE)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f'comparison_{slide_id}.png'))
        plt.close(fig)

if __name__ == '__main__':
    main()