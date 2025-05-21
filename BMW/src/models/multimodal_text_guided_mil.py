# src/models/multimodal_text_guided_mil.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest 

# 尝试从当前目录或定义的路径导入您的真实模块
REAL_MODULES_LOADED = False
try:
    from .attention_layers import SelfAttentionLayer, CrossAttentionLayer
    from .mil_aggregators import AttentionMIL, MILAttentionAggregator 
    print("步骤 1/2 成功: 尝试相对导入真实模块成功。")
    REAL_MODULES_LOADED = True
except ImportError as e_rel:
    print(f"步骤 1/2 失败: 相对导入失败: {e_rel}。尝试绝对导入...")
    try:
        from BMW.src.models.attention_layers import SelfAttentionLayer, CrossAttentionLayer
        from BMW.src.models.mil_aggregators import AttentionMIL, MILAttentionAggregator
        print("步骤 2/2 成功: 尝试绝对导入真实模块成功。")
        REAL_MODULES_LOADED = True
    except ImportError as e_abs:
        print(f"步骤 2/2 失败: 绝对导入也失败: {e_abs}。")
        raise ImportError(
            "关键自定义模块未能加载。请检查 PYTHONPATH、__init__.py 文件以及运行命令的方式。\n"
            f"相对导入错误: {e_rel}\n绝对导入错误: {e_abs}"
        )

if not REAL_MODULES_LOADED:
    raise RuntimeError("真实模块加载状态不一致，未能成功加载真实模块。")

print("确认：将使用已导入的真实模块。后续不应出现'占位符...初始化'信息。")

# NaN/Inf 检查辅助函数
def check_tensor_nan_inf(tensor, name="Tensor", batch_item_idx=None, critical=True, print_content=False):
    prefix = "CRITICAL_NaN_CHECK" if critical else "DEBUG_NaN_CHECK"
    loc_info = f" (Batch Item {batch_item_idx})" if batch_item_idx is not None else ""
    
    if tensor is None:
        return False 

    try:
        # 确保张量在CPU上进行检查，以避免CUDA同步问题和潜在的设备特定行为
        tensor_cpu = tensor.detach().cpu()
        has_nan = torch.isnan(tensor_cpu).any()
        has_inf = torch.isinf(tensor_cpu).any()
    except TypeError: 
        return False
    except RuntimeError as e: # 例如，在某些情况下，尝试 .cpu() 可能会出错
        print(f"ERROR_NaN_CHECK: Could not move tensor {name} to CPU for NaN/Inf check: {e}")
        return False # 无法检查，假设没有问题以继续

    if has_nan or has_inf:
        issues = []
        if has_nan: issues.append("NaN")
        if has_inf: issues.append("Inf")
        print(f"{prefix}: {name}{loc_info} contains {' and '.join(issues)}! Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}")
        if print_content:
            print(f"  {name} values (on CPU): {tensor_cpu}") 
        # import pdb; pdb.set_trace() 
        return True
    return False

class MultimodalTextGuidedMIL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_feature_dim = config.patch_feature_dim
        self.text_feature_dim = config.text_feature_dim
        self.num_classes = config.num_classes
        self.window_patch_rows = config.model_params.window_params.patch_rows
        self.window_patch_cols = config.model_params.window_params.patch_cols
        self.patches_per_window = self.window_patch_rows * self.window_patch_cols
        self.stride_rows = config.model_params.window_params.stride_rows
        self.stride_cols = config.model_params.window_params.stride_cols
        self.num_selected_windows = config.model_params.window_params.num_selected_windows
        self.pre_similarity_window_agg_type = config.model_params.window_params.pre_similarity_window_agg_type
        self.similarity_projection_dim = config.model_params.similarity_projection_dim
        
        self.text_proj_sim = nn.Linear(self.text_feature_dim, self.similarity_projection_dim)
        self.patch_proj_sim = nn.Linear(self.patch_feature_dim, self.similarity_projection_dim)

        if self.pre_similarity_window_agg_type == 'attention_light':
            self.light_window_aggregator = AttentionMIL( 
                input_dim=self.patch_feature_dim,
                hidden_dim=config.model_params.window_params.light_agg_D,
                dropout_rate=config.model_params.window_params.light_agg_dropout,
                output_dim=self.patch_feature_dim 
            )
        elif self.pre_similarity_window_agg_type == 'mean':
            def robust_mean_agg(x, mask): # x: (B, N, D), mask: (B, N)
                if mask is None:
                    check_tensor_nan_inf(x, "x in robust_mean_agg (mask is None)")
                    return x.mean(dim=1)

                float_mask = mask.unsqueeze(-1).float()
                num_valid_patches = float_mask.sum(dim=1) 

                masked_x = x * float_mask 
                sum_feats = masked_x.sum(dim=1) 
                
                safe_num_valid_patches = num_valid_patches + 1e-8 
                aggregated_repr = sum_feats / safe_num_valid_patches
                
                if torch.isnan(aggregated_repr).any():
                    aggregated_repr = torch.nan_to_num(aggregated_repr, nan=0.0, posinf=0.0, neginf=0.0)
                return aggregated_repr
            self.light_window_aggregator = robust_mean_agg

        elif self.pre_similarity_window_agg_type == 'max':
             self.light_window_aggregator = lambda x, mask: x.masked_fill(~mask.unsqueeze(-1).bool(), -1e9).max(dim=1)[0] if mask is not None and mask.any() else x.max(dim=1)[0]
        else:
            raise ValueError(f"不支持的 pre_similarity_window_agg_type: {self.pre_similarity_window_agg_type}")

        self.window_self_attention = SelfAttentionLayer(
            embed_dim=self.patch_feature_dim, 
            num_heads=config.model_params.self_attn_heads,
            dropout=config.model_params.self_attn_dropout
        )
        
        self.window_mil_output_dim = config.model_params.window_mil_output_dim
        self.window_mil_aggregator = AttentionMIL(
            input_dim=self.patch_feature_dim, 
            hidden_dim=config.model_params.window_mil_D, 
            dropout_rate=config.model_params.window_mil_dropout, 
            output_dim=self.window_mil_output_dim
        )
        
        self.final_image_feature_dim = config.model_params.final_image_feature_dim
        self.inter_window_aggregator = AttentionMIL(
            input_dim=self.window_mil_output_dim,
            hidden_dim=config.model_params.inter_window_mil_D, 
            dropout_rate=config.model_params.inter_window_mil_dropout, 
            output_dim=self.final_image_feature_dim
        )
        
        self.cross_attention_output_dim = self.final_image_feature_dim 
        self.cross_attention = CrossAttentionLayer( 
            query_dim=self.final_image_feature_dim,
            key_dim=self.text_feature_dim,
            embed_dim=self.final_image_feature_dim, 
            num_heads=config.model_params.cross_attn_heads,
            dropout=config.model_params.cross_attn_dropout
        )
        
        self.fused_feature_dim = self.final_image_feature_dim + self.cross_attention_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_feature_dim, config.model_params.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.model_params.classifier_dropout),
            nn.Linear(config.model_params.classifier_hidden_dim, self.num_classes)
        )

    def _generate_candidate_spatial_windows(self, all_patch_features_wsi, all_patch_grid_indices_wsi, wsi_grid_shape, batch_item_idx_for_debug=None):
        if check_tensor_nan_inf(all_patch_features_wsi, "all_patch_features_wsi (in _generate_candidate_spatial_windows)", batch_item_idx_for_debug): return [],[]
        
        max_r, max_c = wsi_grid_shape[0].item(), wsi_grid_shape[1].item()
        candidate_windows_feats_list = []
        candidate_windows_masks_list = []
        
        if all_patch_grid_indices_wsi.numel() == 0:
             return [], []
        try:
            coord_to_idx_map = {tuple(coord.tolist()): i for i, coord in enumerate(all_patch_grid_indices_wsi)}
        except Exception as e:
            print(f"ERROR_NaN_CHECK: Failed to create coord_to_idx_map for batch item {batch_item_idx_for_debug}: {e}")
            return [], []

        eff_window_rows = min(self.window_patch_rows, max_r)
        eff_window_cols = min(self.window_patch_cols, max_c)

        if eff_window_rows <= 0 or eff_window_cols <= 0:
            return [], []

        for r_start in range(0, max_r - eff_window_rows + 1, self.stride_rows):
            for c_start in range(0, max_c - eff_window_cols + 1, self.stride_cols):
                current_window_patch_indices = []
                for r_offset in range(eff_window_rows): 
                    for c_offset in range(eff_window_cols):
                        abs_r, abs_c = r_start + r_offset, c_start + c_offset
                        if (abs_r, abs_c) in coord_to_idx_map:
                            current_window_patch_indices.append(coord_to_idx_map[(abs_r, abs_c)])
                
                if len(current_window_patch_indices) > 0:
                    window_feats = all_patch_features_wsi[current_window_patch_indices]
                    if check_tensor_nan_inf(window_feats, f"window_feats (r{r_start}c{c_start})", batch_item_idx_for_debug): continue 

                    num_actual_patches = window_feats.shape[0]
                    padded_window_feats = torch.zeros(self.patches_per_window, self.patch_feature_dim,
                                                      device=all_patch_features_wsi.device, dtype=all_patch_features_wsi.dtype)
                    window_mask = torch.zeros(self.patches_per_window, device=all_patch_features_wsi.device, dtype=torch.bool)
                    
                    num_to_fill = min(num_actual_patches, self.patches_per_window)
                    if num_to_fill > 0: 
                        padded_window_feats[:num_to_fill] = window_feats[:num_to_fill]
                        window_mask[:num_to_fill] = True
                    
                    candidate_windows_feats_list.append(padded_window_feats)
                    candidate_windows_masks_list.append(window_mask)
        
        return candidate_windows_feats_list, candidate_windows_masks_list

    def _select_text_guided_windows(self, all_patch_features_wsi, text_feat_wsi, all_patch_grid_indices_wsi, wsi_grid_shape, batch_item_idx_for_debug=None):
        # Helper to return zero tensors on error or no valid data
        def return_zeros_for_select():
            dev = all_patch_features_wsi.device
            dtype = all_patch_features_wsi.dtype
            feats = torch.zeros(self.num_selected_windows, self.patches_per_window, self.patch_feature_dim, device=dev, dtype=dtype)
            masks = torch.zeros(self.num_selected_windows, self.patches_per_window, device=dev, dtype=torch.bool)
            return feats, masks

        if check_tensor_nan_inf(all_patch_features_wsi, "all_patch_features_wsi (in _select_text_guided_windows)", batch_item_idx_for_debug): return return_zeros_for_select()
        if check_tensor_nan_inf(text_feat_wsi, "text_feat_wsi (in _select_text_guided_windows)", batch_item_idx_for_debug): return return_zeros_for_select()

        candidate_windows_feats_list, candidate_windows_masks_list = \
            self._generate_candidate_spatial_windows(all_patch_features_wsi, all_patch_grid_indices_wsi, wsi_grid_shape, batch_item_idx_for_debug)

        if not candidate_windows_feats_list:
            return return_zeros_for_select()

        aggregated_candidate_reprs = []
        valid_candidate_indices = [] 
        for i, window_feats in enumerate(candidate_windows_feats_list):
            window_mask = candidate_windows_masks_list[i]
            if not window_mask.any(): continue
            
            if check_tensor_nan_inf(window_feats, f"window_feats to light_agg (candidate {i})", batch_item_idx_for_debug): continue
            current_window_feats_unsqueezed = window_feats.unsqueeze(0)
            current_mask_unsqueezed = window_mask.unsqueeze(0)

            if self.pre_similarity_window_agg_type == 'attention_light':
                agg_repr_tuple = self.light_window_aggregator(current_window_feats_unsqueezed, instance_mask=current_mask_unsqueezed)
                agg_repr = agg_repr_tuple[0] 
            else: 
                agg_repr = self.light_window_aggregator(current_window_feats_unsqueezed, current_mask_unsqueezed)
            
            if check_tensor_nan_inf(agg_repr, f"agg_repr from light_agg (candidate {i})", batch_item_idx_for_debug): continue
            aggregated_candidate_reprs.append(agg_repr.squeeze(0))
            valid_candidate_indices.append(i)

        if not aggregated_candidate_reprs:
             return return_zeros_for_select()

        stacked_candidate_reprs = torch.stack(aggregated_candidate_reprs)
        if check_tensor_nan_inf(stacked_candidate_reprs, "stacked_candidate_reprs", batch_item_idx_for_debug): return return_zeros_for_select()

        proj_text_feat = self.text_proj_sim(text_feat_wsi)
        if check_tensor_nan_inf(proj_text_feat, "proj_text_feat", batch_item_idx_for_debug): return return_zeros_for_select()
        
        proj_candidate_reprs = self.patch_proj_sim(stacked_candidate_reprs)
        if check_tensor_nan_inf(proj_candidate_reprs, "proj_candidate_reprs", batch_item_idx_for_debug): return return_zeros_for_select()

        text_norm = torch.linalg.norm(proj_text_feat.unsqueeze(0), dim=-1) 
        candidate_norms = torch.linalg.norm(proj_candidate_reprs, dim=-1)
        if torch.any(text_norm < 1e-7): 
            print(f"WARNING_NaN_CHECK: proj_text_feat norm is very small: {text_norm.item()} for batch item {batch_item_idx_for_debug}")
        if torch.any(candidate_norms < 1e-7):
            small_norms_indices = (candidate_norms < 1e-7).nonzero(as_tuple=True)[0]
            print(f"WARNING_NaN_CHECK: {small_norms_indices.size(0)} proj_candidate_reprs norms are very small for batch item {batch_item_idx_for_debug} at indices {small_norms_indices.tolist()}. Smallest: {candidate_norms.min().item() if candidate_norms.numel() > 0 else 'N/A'}")
            # proj_candidate_reprs[small_norms_indices] = torch.rand_like(proj_candidate_reprs[small_norms_indices]) * 1e-6 # Add small random noise if norms are zero

        similarity_scores = F.cosine_similarity(proj_text_feat.unsqueeze(0), proj_candidate_reprs, dim=1)
        if check_tensor_nan_inf(similarity_scores, "similarity_scores", batch_item_idx_for_debug): 
            # print(f"  proj_text_feat for NaN scores: {proj_text_feat}")
            # print(f"  proj_candidate_reprs for NaN scores (first few): {proj_candidate_reprs[:min(3, proj_candidate_reprs.shape[0])]}")
            return return_zeros_for_select()
        
        num_valid_candidates = similarity_scores.shape[0]
        num_to_select = min(self.num_selected_windows, num_valid_candidates)
        
        final_selected_feats_list = []
        final_selected_masks_list = []

        if num_to_select > 0:
            if not (torch.isnan(similarity_scores).any() or torch.isinf(similarity_scores).any()):
                _, top_k_relative_indices = torch.topk(similarity_scores, k=num_to_select, dim=0)
                top_k_absolute_indices = [valid_candidate_indices[i] for i in top_k_relative_indices.tolist()]
                for idx in top_k_absolute_indices:
                    final_selected_feats_list.append(candidate_windows_feats_list[idx])
                    final_selected_masks_list.append(candidate_windows_masks_list[idx])

        if final_selected_feats_list:
            padded_selected_feats = torch.stack(final_selected_feats_list, dim=0)
            padded_selected_masks = torch.stack(final_selected_masks_list, dim=0)
        else: 
            padded_selected_feats = torch.zeros(0, self.patches_per_window, self.patch_feature_dim, device=all_patch_features_wsi.device, dtype=all_patch_features_wsi.dtype)
            padded_selected_masks = torch.zeros(0, self.patches_per_window, device=all_patch_features_wsi.device, dtype=torch.bool)
        
        num_padding_windows = self.num_selected_windows - padded_selected_feats.shape[0]
        if num_padding_windows > 0:
            padding_f = torch.zeros(num_padding_windows, self.patches_per_window, self.patch_feature_dim, device=all_patch_features_wsi.device, dtype=all_patch_features_wsi.dtype)
            padding_m = torch.zeros(num_padding_windows, self.patches_per_window, device=all_patch_features_wsi.device, dtype=torch.bool)
            if padded_selected_feats.numel() > 0 : 
                padded_selected_feats = torch.cat([padded_selected_feats, padding_f], dim=0)
                padded_selected_masks = torch.cat([padded_selected_masks, padding_m], dim=0)
            else: 
                padded_selected_feats = padding_f
                padded_selected_masks = padding_m
        
        return padded_selected_feats, padded_selected_masks

    def forward(self, image_patch_features_batch, patch_grid_indices_batch, text_feat_batch, grid_shapes_batch, original_patch_coordinates_batch=None, patch_mask_batch=None):
        def nan_return_val():
            return torch.full((image_patch_features_batch.shape[0], self.num_classes), float('nan'), device=image_patch_features_batch.device, dtype=image_patch_features_batch.dtype)

        if check_tensor_nan_inf(image_patch_features_batch, "image_patch_features_batch @ FORWARD_ENTRY"): return nan_return_val()
        if check_tensor_nan_inf(text_feat_batch, "text_feat_batch @ FORWARD_ENTRY"): return nan_return_val()
        
        batch_size = image_patch_features_batch.shape[0]
        all_selected_windows_feats_b = []
        all_selected_windows_masks_b = []

        for i in range(batch_size):
            current_patch_feats = image_patch_features_batch[i]
            current_grid_indices = patch_grid_indices_batch[i]
            current_text_feat = text_feat_batch[i]
            current_grid_shape = grid_shapes_batch[i]
            
            active_patch_feats = current_patch_feats
            active_grid_indices = current_grid_indices
            
            num_original_patches = current_patch_feats.shape[0]
            if patch_mask_batch is not None and patch_mask_batch[i] is not None:
                current_patch_mask = patch_mask_batch[i]
                if current_patch_mask.shape[0] != num_original_patches:
                    len_to_use = min(current_patch_mask.shape[0], num_original_patches)
                    valid_patches_mask_i = current_patch_mask[:len_to_use].bool()
                    active_patch_feats = current_patch_feats[:len_to_use][valid_patches_mask_i]
                    active_grid_indices = current_grid_indices[:len_to_use][valid_patches_mask_i]
                else:
                    valid_patches_mask_i = current_patch_mask.bool()
                    active_patch_feats = current_patch_feats[valid_patches_mask_i]
                    active_grid_indices = current_grid_indices[valid_patches_mask_i]
            
            if check_tensor_nan_inf(active_patch_feats, f"active_patch_feats (batch item {i})"): return nan_return_val()

            if active_patch_feats.shape[0] == 0:
                s_feats = torch.zeros(self.num_selected_windows, self.patches_per_window, self.patch_feature_dim, device=current_patch_feats.device, dtype=current_patch_feats.dtype)
                s_mask = torch.zeros(self.num_selected_windows, self.patches_per_window, device=current_patch_feats.device, dtype=torch.bool)
            else:
                s_feats, s_mask = self._select_text_guided_windows(
                    active_patch_feats, current_text_feat, active_grid_indices, current_grid_shape, batch_item_idx_for_debug=i
                )
            
            if check_tensor_nan_inf(s_feats, f"s_feats from _select_text_guided_windows (batch item {i})"): return nan_return_val()
            all_selected_windows_feats_b.append(s_feats)
            all_selected_windows_masks_b.append(s_mask)

        selected_windows_feats = torch.stack(all_selected_windows_feats_b, dim=0)
        if check_tensor_nan_inf(selected_windows_feats, "selected_windows_feats (after stack)"): return nan_return_val()
        
        selected_windows_mask = torch.stack(all_selected_windows_masks_b, dim=0)
        
        k_w = self.num_selected_windows
        # proc_windows_feats: (B * k_w, patches_per_window, D_patch)
        # proc_windows_mask:  (B * k_w, patches_per_window)
        proc_windows_feats = selected_windows_feats.reshape(batch_size * k_w, self.patches_per_window, self.patch_feature_dim)
        proc_windows_mask = selected_windows_mask.reshape(batch_size * k_w, self.patches_per_window)
        
        if check_tensor_nan_inf(proc_windows_feats, "proc_windows_feats (INPUT to self_attn)"): return nan_return_val()
        
        # --- MODIFICATION FOR HANDLING FULLY MASKED WINDOWS IN SELF-ATTENTION ---
        # key_padding_mask for MultiheadAttention: True for positions to be IGNORED
        current_key_padding_mask = ~proc_windows_mask.bool() if proc_windows_mask is not None else None

        # Initialize attended_patch_feats with zeros
        attended_patch_feats = torch.zeros_like(proc_windows_feats)

        if current_key_padding_mask is not None:
            # Identify windows that are NOT fully masked (i.e., have at least one valid patch)
            # A window is NOT fully masked if its key_padding_mask row is NOT all True
            not_fully_masked_window_indices = (~current_key_padding_mask.all(dim=1)).nonzero(as_tuple=True)[0]
            
            # Identify windows that ARE fully masked
            fully_masked_window_indices = current_key_padding_mask.all(dim=1).nonzero(as_tuple=True)[0]
            if fully_masked_window_indices.numel() > 0:
                 print(f"DEBUG_NaN_CHECK: {fully_masked_window_indices.numel()} windows are fully masked and will have zero output from self-attention.")
        else: # No mask provided, so no windows are fully masked by definition of the mask
            not_fully_masked_window_indices = torch.arange(proc_windows_feats.shape[0], device=proc_windows_feats.device)
            # fully_masked_window_indices is empty

        if not_fully_masked_window_indices.numel() > 0:
            # print(f"DEBUG_NaN_CHECK: Processing {not_fully_masked_window_indices.numel()} non-fully-masked windows in self-attention.")
            feats_to_attend = proc_windows_feats[not_fully_masked_window_indices]
            mask_for_attention = current_key_padding_mask[not_fully_masked_window_indices] if current_key_padding_mask is not None else None
            
            if check_tensor_nan_inf(feats_to_attend, "feats_to_attend (subset for self_attn)"): return nan_return_val()

            output_from_attention = self.window_self_attention(
                feats_to_attend,
                key_padding_mask=mask_for_attention
            )
            if check_tensor_nan_inf(output_from_attention, "output_from_attention (from non-fully-masked self_attn)"): return nan_return_val()
            attended_patch_feats.index_copy_(0, not_fully_masked_window_indices, output_from_attention)
        # For fully_masked_window_indices, attended_patch_feats remains zeros, which is a safe default.
        # --- END MODIFICATION ---

        if check_tensor_nan_inf(attended_patch_feats, "attended_patch_feats (AFTER selective self_attn)"): return nan_return_val()
        
        aggregated_window_reprs, _ = self.window_mil_aggregator(
            attended_patch_feats,
            instance_mask=proc_windows_mask 
        ) 
        if check_tensor_nan_inf(aggregated_window_reprs, "aggregated_window_reprs (from window_mil_aggregator)"): return nan_return_val()
        
        aggregated_window_reprs = aggregated_window_reprs.view(batch_size, k_w, self.window_mil_output_dim)
        if check_tensor_nan_inf(aggregated_window_reprs, "aggregated_window_reprs (reshaped for inter_window_agg)"): return nan_return_val()
        
        inter_window_mask = selected_windows_mask.any(dim=2) 
        if check_tensor_nan_inf(inter_window_mask.float(), "inter_window_mask", critical=False): return nan_return_val()

        final_image_repr, _ = self.inter_window_aggregator(
            aggregated_window_reprs,
            instance_mask=inter_window_mask
        ) 
        if check_tensor_nan_inf(final_image_repr, "final_image_repr (from inter_window_aggregator)"): return nan_return_val()
        
        final_image_repr_seq = final_image_repr.unsqueeze(1)
        text_feat_batch_seq = text_feat_batch.unsqueeze(1)
        
        fused_representation = self.cross_attention(
            query=final_image_repr_seq,
            key_value=text_feat_batch_seq
        ) 
        if check_tensor_nan_inf(fused_representation, "fused_representation (from cross_attention)"): return nan_return_val()
        
        fused_representation = fused_representation.squeeze(1)
        
        final_multimodal_feat = torch.cat([final_image_repr, fused_representation], dim=-1)
        if check_tensor_nan_inf(final_multimodal_feat, "final_multimodal_feat (after cat)"): return nan_return_val()
        
        logits = self.classifier(final_multimodal_feat)
        if check_tensor_nan_inf(logits, "logits (FINAL OUTPUT from classifier)"):
            if not (torch.isnan(final_multimodal_feat).any() or torch.isinf(final_multimodal_feat).any()):
                 print(f"  final_multimodal_feat (input to classifier) min: {final_multimodal_feat.min().item()}, max: {final_multimodal_feat.max().item()}, mean: {final_multimodal_feat.mean().item()}")
        
        return logits

# --- 简单的配置类 (保持不变) ---
class SimpleConfig:
    def __init__(self, **kwargs):
        self.patch_feature_dim = kwargs.get('patch_feature_dim', 1024)
        self.text_feature_dim = kwargs.get('text_feature_dim', 1024)
        self.num_classes = kwargs.get('num_classes', 2)
        model_params_config = kwargs.get('model_params', {})
        if not isinstance(model_params_config, SimpleConfigModelParams):
             model_params_config = SimpleConfigModelParams(**model_params_config)
        self.model_params = model_params_config
    def get(self, key, default=None): return getattr(self, key, default)

class SimpleConfigModelParams:
    def __init__(self, **kwargs):
        self.similarity_projection_dim = kwargs.get('similarity_projection_dim', 256)
        window_params_config = kwargs.get('window_params', {})
        if not isinstance(window_params_config, SimpleConfigWindowParams):
            window_params_config = SimpleConfigWindowParams(**window_params_config)
        self.window_params = window_params_config
        self.self_attn_heads = kwargs.get('self_attn_heads', 4)
        self.self_attn_dropout = kwargs.get('self_attn_dropout', 0.1)
        self.window_mil_output_dim = kwargs.get('window_mil_output_dim', 512)
        self.window_mil_D = kwargs.get('window_mil_D', 128) 
        self.window_mil_dropout = kwargs.get('window_mil_dropout', 0.1) 
        self.final_image_feature_dim = kwargs.get('final_image_feature_dim', 512)
        self.inter_window_mil_D = kwargs.get('inter_window_mil_D', 64) 
        self.inter_window_mil_dropout = kwargs.get('inter_window_mil_dropout', 0.1) 
        self.cross_attn_heads = kwargs.get('cross_attn_heads', 4)
        self.cross_attn_dropout = kwargs.get('cross_attn_dropout', 0.1)
        self.classifier_hidden_dim = kwargs.get('classifier_hidden_dim', 256)
        self.classifier_dropout = kwargs.get('classifier_dropout', 0.25)
    def get(self, key, default=None): return getattr(self, key, default)

class SimpleConfigWindowParams:
    def __init__(self, **kwargs):
        self.patch_rows = kwargs.get('patch_rows', 3)
        self.patch_cols = kwargs.get('patch_cols', 3)
        self.stride_rows = kwargs.get('stride_rows', 3)
        self.stride_cols = kwargs.get('stride_cols', 3)
        self.num_selected_windows = kwargs.get('num_selected_windows', 5)
        self.pre_similarity_window_agg_type = kwargs.get('pre_similarity_window_agg_type', 'mean')
        self.light_agg_D = kwargs.get('light_agg_D', 64) 
        self.light_agg_dropout = kwargs.get('light_agg_dropout', 0.1)

# --- 单元测试部分 (保持不变) ---
class TestMultimodalTextGuidedMIL(unittest.TestCase):
    def setUp(self):
        self.config_dict = {
            'patch_feature_dim': 64, 
            'text_feature_dim': 32,
            'num_classes': 2,
            'model_params': {
                'similarity_projection_dim': 16,
                'window_params': {
                    'patch_rows': 2, 'patch_cols': 2, 
                    'stride_rows': 1, 'stride_cols': 1,
                    'num_selected_windows': 3,        
                    'pre_similarity_window_agg_type': 'mean', 
                    'light_agg_D': 8, 
                    'light_agg_dropout': 0.0, 
                },
                'self_attn_heads': 2, 'self_attn_dropout': 0.0,
                'window_mil_output_dim': 24, 
                'window_mil_D': 12, 
                'window_mil_dropout': 0.0, 
                'final_image_feature_dim': 20, 
                'inter_window_mil_D': 10, 
                'inter_window_mil_dropout': 0.0, 
                'cross_attn_heads': 2, 'cross_attn_dropout': 0.0,
                'classifier_hidden_dim': 16, 'classifier_dropout': 0.0,
            }
        }
        self.config = SimpleConfig(**self.config_dict)
        self.model = MultimodalTextGuidedMIL(self.config)
        self.model.eval() 

        self.patch_dim = self.config.patch_feature_dim
        self.text_dim = self.config.text_feature_dim
        self.N_total_patches = 25 
        self.all_patch_features_wsi = torch.randn(self.N_total_patches, self.patch_dim)
        grid_r, grid_c = torch.meshgrid(torch.arange(5), torch.arange(5), indexing='ij')
        self.all_patch_grid_indices_wsi = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=1)
        self.text_feat_wsi = torch.randn(self.text_dim)
        self.wsi_grid_shape = torch.tensor([5, 5], dtype=torch.long)

    def test_01_generate_candidate_spatial_windows(self):
        print("\n测试: _generate_candidate_spatial_windows")
        self.model.window_patch_rows = self.config.model_params.window_params.patch_rows
        self.model.window_patch_cols = self.config.model_params.window_params.patch_cols
        self.model.patches_per_window = self.model.window_patch_rows * self.model.window_patch_cols
        self.model.stride_rows = self.config.model_params.window_params.stride_rows
        self.model.stride_cols = self.config.model_params.window_params.stride_cols

        feats_list, masks_list = self.model._generate_candidate_spatial_windows(
            self.all_patch_features_wsi,
            self.all_patch_grid_indices_wsi,
            self.wsi_grid_shape,
            batch_item_idx_for_debug="test_01"
        )
        
        expected_rows = (self.wsi_grid_shape[0].item() - self.model.window_patch_rows) // self.model.stride_rows + 1
        expected_cols = (self.wsi_grid_shape[1].item() - self.model.window_patch_cols) // self.model.stride_cols + 1
        expected_num_candidates = expected_rows * expected_cols

        self.assertEqual(len(feats_list), expected_num_candidates)
        self.assertEqual(len(masks_list), expected_num_candidates)
        if feats_list:
            self.assertEqual(feats_list[0].shape, (self.model.patches_per_window, self.patch_dim))
            self.assertEqual(masks_list[0].shape, (self.model.patches_per_window,))
            self.assertTrue(masks_list[0].all()) 
        print(f"  生成了 {len(feats_list)} 个候选窗口 (预期 {expected_num_candidates})。")

    def test_02_select_text_guided_windows(self):
        print("\n测试: _select_text_guided_windows")
        self.model.window_patch_rows = self.config.model_params.window_params.patch_rows
        self.model.window_patch_cols = self.config.model_params.window_params.patch_cols
        self.model.patches_per_window = self.model.window_patch_rows * self.model.window_patch_cols
        self.model.stride_rows = self.config.model_params.window_params.stride_rows
        self.model.stride_cols = self.config.model_params.window_params.stride_cols
        self.model.num_selected_windows = self.config.model_params.window_params.num_selected_windows
        
        sel_feats, sel_masks = self.model._select_text_guided_windows(
            self.all_patch_features_wsi,
            self.text_feat_wsi,
            self.all_patch_grid_indices_wsi,
            self.wsi_grid_shape,
            batch_item_idx_for_debug="test_02"
        )
        self.assertEqual(sel_feats.shape, (self.model.num_selected_windows, self.model.patches_per_window, self.patch_dim))
        self.assertEqual(sel_masks.shape, (self.model.num_selected_windows, self.model.patches_per_window))
        print(f"  选定了 {sel_feats.shape[0]} 个窗口。")

    def test_03_forward_pass_single_sample_no_padding(self):
        print("\n测试: forward pass (单样本, 无需padding)")
        bs = 1
        img_feats_b = self.all_patch_features_wsi.unsqueeze(0) 
        grid_idx_b = self.all_patch_grid_indices_wsi.unsqueeze(0) 
        txt_feat_b = self.text_feat_wsi.unsqueeze(0) 
        grid_shape_b = self.wsi_grid_shape.unsqueeze(0) 
        
        logits = self.model(img_feats_b, grid_idx_b, txt_feat_b, grid_shape_b, patch_mask_batch=None)
        self.assertEqual(logits.shape, (bs, self.config.num_classes))
        self.assertFalse(torch.isnan(logits).any(), "Logits should not be NaN in test_03")
        print(f"  单样本前向传播成功，Logits 形状: {logits.shape}")

    def test_04_forward_pass_batch_with_padding_mask(self):
        print("\n测试: forward pass (批处理, 带padding mask)")
        bs = 2
        max_patches_in_batch = 30 
        
        s1_feats_actual = self.all_patch_features_wsi 
        s1_grid_idx_actual = self.all_patch_grid_indices_wsi
        s1_grid_shape_actual = self.wsi_grid_shape
        s1_num_actual_patches = s1_feats_actual.shape[0]
        
        s2_N_patches = 9
        s2_patch_dim = self.config.patch_feature_dim
        s2_feats_actual = torch.randn(s2_N_patches, s2_patch_dim)
        s2_grid_r, s2_grid_c = torch.meshgrid(torch.arange(3), torch.arange(3), indexing='ij')
        s2_grid_idx_actual = torch.stack([s2_grid_r.flatten(), s2_grid_c.flatten()], dim=1)
        s2_grid_shape_actual = torch.tensor([3, 3], dtype=torch.long)

        img_feats_b = torch.zeros(bs, max_patches_in_batch, self.patch_dim)
        grid_idx_b = torch.zeros(bs, max_patches_in_batch, 2, dtype=torch.long) 
        patch_mask_b = torch.zeros(bs, max_patches_in_batch, dtype=torch.bool) 
        
        img_feats_b[0, :s1_num_actual_patches] = s1_feats_actual
        grid_idx_b[0, :s1_num_actual_patches] = s1_grid_idx_actual
        patch_mask_b[0, :s1_num_actual_patches] = True
        
        img_feats_b[1, :s2_N_patches] = s2_feats_actual
        grid_idx_b[1, :s2_N_patches] = s2_grid_idx_actual
        patch_mask_b[1, :s2_N_patches] = True
        
        txt_feat_b = torch.randn(bs, self.text_dim)
        grid_shape_b = torch.stack([s1_grid_shape_actual, s2_grid_shape_actual], dim=0)
        
        logits = self.model(img_feats_b, grid_idx_b, txt_feat_b, grid_shape_b, patch_mask_batch=patch_mask_b)
        self.assertEqual(logits.shape, (bs, self.config.num_classes))
        self.assertFalse(torch.isnan(logits).any(), "Logits should not be NaN in test_04")
        print(f"  批处理前向传播成功，Logits 形状: {logits.shape}")

    def test_05_empty_active_patches(self):
        print("\n测试: forward pass (一个样本完全被掩码)")
        bs = 1
        max_patches_in_batch = self.N_total_patches
        img_feats_b = self.all_patch_features_wsi.unsqueeze(0)
        grid_idx_b = self.all_patch_grid_indices_wsi.unsqueeze(0)
        txt_feat_b = self.text_feat_wsi.unsqueeze(0)
        grid_shape_b = self.wsi_grid_shape.unsqueeze(0)
        patch_mask_b = torch.zeros(bs, max_patches_in_batch, dtype=torch.bool) 

        logits = self.model(img_feats_b, grid_idx_b, txt_feat_b, grid_shape_b, patch_mask_batch=patch_mask_b)
        self.assertEqual(logits.shape, (bs, self.config.num_classes))
        self.assertFalse(torch.isnan(logits).any(), "Logits should not be NaN when a sample is fully masked")
        print(f"  完全掩码样本前向传播成功，Logits 形状: {logits.shape}")

if __name__ == '__main__':
    print("开始 MultimodalTextGuidedMIL 模型组件和前向传播的单元测试...")
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMultimodalTextGuidedMIL))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    print("\n所有单元测试执行完毕。")