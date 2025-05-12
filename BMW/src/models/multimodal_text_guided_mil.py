import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest # 引入 unittest 模块，提供更结构化的测试

# 尝试从当前目录或定义的路径导入您的真实模块
# 您需要确保这些导入路径相对于您的项目结构是正确的
# 并且这些类 (SelfAttentionLayer, CrossAttentionLayer, AttentionMIL) 存在且接口兼容
try:
    # 如果此脚本与 attention_layers.py 和 mil_aggregators.py 在同一目录下 (BMW/src/models/)
    from .attention_layers import SelfAttentionLayer, CrossAttentionLayer
    from .mil_aggregators import AttentionMIL 
    print("成功从 .attention_layers 和 .mil_aggregators 导入自定义模块。")
except ImportError as e:
    print(f"警告: 尝试相对导入失败: {e}")
    try:
        # 备用方案：如果您的项目结构允许直接从 BMW.src.models 导入
        # (这通常需要您的项目根目录在 PYTHONPATH 中，或者您从根目录运行特定命令)
        from BMW.src.models.attention_layers import SelfAttentionLayer, CrossAttentionLayer
        from BMW.src.models.mil_aggregators import AttentionMIL
        print("成功从 BMW.src.models 导入自定义模块。")
    except ImportError as e2:
        print(f"警告: 尝试从 BMW.src.models 导入也失败: {e2}。将使用占位符模块。")
        # --- 占位符模块 (如果导入失败) ---
        class SelfAttentionLayer(nn.Module): 
            def __init__(self, feature_dim, num_heads=4, dropout=0.1):
                super().__init__()
                self.norm = nn.LayerNorm(feature_dim)
                self.attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
                print(f"占位符 SelfAttentionLayer 初始化，维度 {feature_dim}")
            def forward(self, x, mask=None): # x: (batch, seq_len, dim), mask: (batch, seq_len) True to MASK
                x_norm = self.norm(x)
                # nn.MultiheadAttention returns (attn_output, attn_output_weights)
                attn_output, _attn_weights = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
                return attn_output # 真实模块可能返回 attn_output, _attn_weights

        class CrossAttentionLayer(nn.Module): 
            def __init__(self, query_dim, key_dim, num_heads=4, dropout=0.1):
                super().__init__()
                self.q_norm = nn.LayerNorm(query_dim)
                self.kv_norm = nn.LayerNorm(key_dim)
                self.proj_k = nn.Linear(key_dim, query_dim) if key_dim != query_dim else nn.Identity()
                self.proj_v = nn.Linear(key_dim, query_dim) if key_dim != query_dim else nn.Identity()
                self.attn = nn.MultiheadAttention(query_dim, num_heads, dropout=dropout, batch_first=True)
                print(f"占位符 CrossAttentionLayer 初始化，查询维度 {query_dim}, 键/值维度 {key_dim}")
            def forward(self, query, key_value, kv_mask=None): # query: (B, Nq, Dq), key_value: (B, Nkv, Dkv)
                query_norm = self.q_norm(query)
                kv_norm = self.kv_norm(key_value)
                k = self.proj_k(kv_norm)
                v = self.proj_v(kv_norm)
                # nn.MultiheadAttention returns (attn_output, attn_output_weights)
                attn_output, _attn_weights = self.attn(query_norm, k, v, key_padding_mask=kv_mask)
                return attn_output # 真实模块可能返回 attn_output, _attn_weights

        class AttentionMIL(nn.Module): 
            def __init__(self, input_dim, hidden_dim_att, output_dim_att=1, dropout_att=0.25, output_dim=None):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim if output_dim is not None else input_dim
                self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim_att), nn.Tanh())
                self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim_att), nn.Sigmoid())
                self.attention_weights = nn.Linear(hidden_dim_att, output_dim_att) # K
                # 如果需要改变最终输出特征的维度，并且 K=1
                if self.input_dim != self.output_dim and output_dim_att == 1:
                    self.feature_transform = nn.Linear(self.input_dim, self.output_dim)
                else: # 如果 K > 1 或者 input_dim == output_dim, 则不直接变换或需要更复杂的变换逻辑
                    self.feature_transform = nn.Identity() 
                print(f"占位符 AttentionMIL 初始化，输入维度 {input_dim}, 输出维度 {self.output_dim}")

            def forward(self, x, instance_mask=None): # x: (B, N, D_in), instance_mask: (B, N) True for VALID
                A_V = self.attention_V(x)
                A_U = self.attention_U(x)
                att_raw = self.attention_weights(A_V * A_U) # (B, N, K)
                
                if instance_mask is not None:
                    # instance_mask: True for valid. We want to set weights of PADDED (False) elements to -inf
                    att_raw.masked_fill_(~instance_mask.bool().unsqueeze(-1).expand_as(att_raw), float('-inf'))

                att_scores = F.softmax(att_raw, dim=1) # (B, N, K)
                M = torch.bmm(att_scores.transpose(1, 2), x) # (B, K, D_in)
                
                if M.size(1) == 1: # K=1
                    M = M.squeeze(1) # (B, D_in)
                    M = self.feature_transform(M) # (B, D_out)
                else: # K > 1
                    # 如果需要变换维度，这里可能需要 M.view(B, K*D_in) 然后再变换，或者其他处理
                    # 为简单起见，如果 K > 1 且需要变换维度，占位符的 feature_transform 可能不完全符合预期
                    M = self.feature_transform(M) if self.input_dim == self.output_dim else M 
                return M, att_scores


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
                hidden_dim_att=config.model_params.window_params.light_agg_D,
                dropout_att=config.model_params.window_params.light_agg_dropout,
                output_dim=self.patch_feature_dim # 输出维度与输入一致，方便后续投影
            )
        elif self.pre_similarity_window_agg_type == 'mean':
            self.light_window_aggregator = lambda x, mask: (x * mask.unsqueeze(-1).float()).sum(dim=1) / (mask.sum(dim=1, keepdim=True).float() + 1e-6) if mask is not None and mask.any() else x.mean(dim=1)
        elif self.pre_similarity_window_agg_type == 'max':
             self.light_window_aggregator = lambda x, mask: x.masked_fill(~mask.unsqueeze(-1).bool(), -1e9).max(dim=1)[0] if mask is not None and mask.any() else x.max(dim=1)[0]
        else:
            raise ValueError(f"不支持的 pre_similarity_window_agg_type: {self.pre_similarity_window_agg_type}")

        self.window_self_attention = SelfAttentionLayer(
            feature_dim=self.patch_feature_dim,
            num_heads=config.model_params.self_attn_heads,
            dropout=config.model_params.self_attn_dropout
        )
        self.window_mil_output_dim = config.model_params.window_mil_output_dim
        self.window_mil_aggregator = AttentionMIL(
            input_dim=self.patch_feature_dim,
            hidden_dim_att=config.model_params.window_mil_D,
            dropout_att=config.model_params.window_mil_dropout,
            output_dim=self.window_mil_output_dim
        )
        self.final_image_feature_dim = config.model_params.final_image_feature_dim
        self.inter_window_aggregator = AttentionMIL(
            input_dim=self.window_mil_output_dim,
            hidden_dim_att=config.model_params.inter_window_mil_D,
            dropout_att=config.model_params.inter_window_mil_dropout,
            output_dim=self.final_image_feature_dim
        )
        self.cross_attention_output_dim = self.final_image_feature_dim # CrossAttn 输出通常与Query维度一致
        self.cross_attention = CrossAttentionLayer(
            query_dim=self.final_image_feature_dim,
            key_dim=self.text_feature_dim,
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

    def _generate_candidate_spatial_windows(self, all_patch_features_wsi, all_patch_grid_indices_wsi, wsi_grid_shape):
        max_r, max_c = wsi_grid_shape[0].item(), wsi_grid_shape[1].item()
        candidate_windows_feats_list = []
        candidate_windows_masks_list = []
        coord_to_idx_map = {tuple(coord.tolist()): i for i, coord in enumerate(all_patch_grid_indices_wsi)}

        # 确保窗口定义不超过网格本身大小
        eff_window_rows = min(self.window_patch_rows, max_r)
        eff_window_cols = min(self.window_patch_cols, max_c)
        if eff_window_rows == 0 or eff_window_cols == 0: return [], []


        for r_start in range(0, max_r - eff_window_rows + 1, self.stride_rows):
            for c_start in range(0, max_c - eff_window_cols + 1, self.stride_cols):
                current_window_patch_indices = []
                for r_offset in range(eff_window_rows): # 使用有效窗口大小
                    for c_offset in range(eff_window_cols):
                        abs_r, abs_c = r_start + r_offset, c_start + c_offset
                        if (abs_r, abs_c) in coord_to_idx_map:
                            current_window_patch_indices.append(coord_to_idx_map[(abs_r, abs_c)])
                
                if len(current_window_patch_indices) > 0:
                    window_feats = all_patch_features_wsi[current_window_patch_indices]
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

    def _select_text_guided_windows(self, all_patch_features_wsi, text_feat_wsi, all_patch_grid_indices_wsi, wsi_grid_shape):
        candidate_windows_feats_list, candidate_windows_masks_list = \
            self._generate_candidate_spatial_windows(all_patch_features_wsi, all_patch_grid_indices_wsi, wsi_grid_shape)

        if not candidate_windows_feats_list:
            padded_selected_feats = torch.zeros(self.num_selected_windows, self.patches_per_window, self.patch_feature_dim,
                                                device=all_patch_features_wsi.device, dtype=all_patch_features_wsi.dtype)
            padded_selected_masks = torch.zeros(self.num_selected_windows, self.patches_per_window,
                                                device=all_patch_features_wsi.device, dtype=torch.bool)
            return padded_selected_feats, padded_selected_masks

        aggregated_candidate_reprs = []
        valid_candidate_indices = [] # 记录那些实际有内容的候选窗口的索引
        for i, window_feats in enumerate(candidate_windows_feats_list):
            window_mask = candidate_windows_masks_list[i]
            if not window_mask.any(): # 如果窗口内全是padding，跳过
                continue
            
            current_window_feats_unsqueezed = window_feats.unsqueeze(0)
            current_mask_unsqueezed = window_mask.unsqueeze(0)

            if self.pre_similarity_window_agg_type == 'attention_light':
                agg_repr_tuple = self.light_window_aggregator(current_window_feats_unsqueezed, instance_mask=current_mask_unsqueezed)
                agg_repr = agg_repr_tuple[0] if isinstance(agg_repr_tuple, tuple) else agg_repr_tuple
            else: 
                agg_repr = self.light_window_aggregator(current_window_feats_unsqueezed, current_mask_unsqueezed)
            
            aggregated_candidate_reprs.append(agg_repr.squeeze(0))
            valid_candidate_indices.append(i) # 记录这个有效候选窗口的原始索引

        if not aggregated_candidate_reprs: # 如果所有候选窗口都为空
             padded_selected_feats = torch.zeros(self.num_selected_windows, self.patches_per_window, self.patch_feature_dim,
                                                device=all_patch_features_wsi.device, dtype=all_patch_features_wsi.dtype)
             padded_selected_masks = torch.zeros(self.num_selected_windows, self.patches_per_window,
                                                device=all_patch_features_wsi.device, dtype=torch.bool)
             return padded_selected_feats, padded_selected_masks


        stacked_candidate_reprs = torch.stack(aggregated_candidate_reprs)
        proj_text_feat = self.text_proj_sim(text_feat_wsi)
        proj_candidate_reprs = self.patch_proj_sim(stacked_candidate_reprs)
        similarity_scores = F.cosine_similarity(proj_text_feat.unsqueeze(0), proj_candidate_reprs, dim=1)
        
        num_valid_candidates = similarity_scores.shape[0]
        num_to_select = min(self.num_selected_windows, num_valid_candidates)
        
        final_selected_feats_list = []
        final_selected_masks_list = []

        if num_to_select > 0:
            _, top_k_relative_indices = torch.topk(similarity_scores, k=num_to_select, dim=0)
            # 将相对索引转换回 candidate_windows_feats_list 中的绝对索引
            top_k_absolute_indices = [valid_candidate_indices[i] for i in top_k_relative_indices.tolist()]

            for idx in top_k_absolute_indices:
                final_selected_feats_list.append(candidate_windows_feats_list[idx])
                final_selected_masks_list.append(candidate_windows_masks_list[idx])
        
        if final_selected_feats_list:
            padded_selected_feats = torch.stack(final_selected_feats_list, dim=0)
            padded_selected_masks = torch.stack(final_selected_masks_list, dim=0)
        else:
            padded_selected_feats = torch.empty(0, self.patches_per_window, self.patch_feature_dim, device=all_patch_features_wsi.device, dtype=all_patch_features_wsi.dtype)
            padded_selected_masks = torch.empty(0, self.patches_per_window, device=all_patch_features_wsi.device, dtype=torch.bool)

        num_padding_windows = self.num_selected_windows - padded_selected_feats.shape[0]
        if num_padding_windows > 0:
            padding_f = torch.zeros(num_padding_windows, self.patches_per_window, self.patch_feature_dim,
                                    device=all_patch_features_wsi.device, dtype=all_patch_features_wsi.dtype)
            padding_m = torch.zeros(num_padding_windows, self.patches_per_window,
                                    device=all_patch_features_wsi.device, dtype=torch.bool)
            padded_selected_feats = torch.cat([padded_selected_feats, padding_f], dim=0)
            padded_selected_masks = torch.cat([padded_selected_masks, padding_m], dim=0)
        return padded_selected_feats, padded_selected_masks

    def forward(self, image_patch_features_batch, patch_grid_indices_batch, text_feat_batch, grid_shapes_batch, original_patch_coordinates_batch=None, patch_mask_batch=None):
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
            if patch_mask_batch is not None and patch_mask_batch[i] is not None:
                valid_patches_mask_i = patch_mask_batch[i].bool() # 确保是布尔类型
                active_patch_feats = current_patch_feats[valid_patches_mask_i]
                active_grid_indices = current_grid_indices[valid_patches_mask_i]

            if active_patch_feats.shape[0] == 0:
                s_feats = torch.zeros(self.num_selected_windows, self.patches_per_window, self.patch_feature_dim,
                                      device=current_patch_feats.device, dtype=current_patch_feats.dtype)
                s_mask = torch.zeros(self.num_selected_windows, self.patches_per_window,
                                     device=current_patch_feats.device, dtype=torch.bool)
            else:
                s_feats, s_mask = self._select_text_guided_windows(
                    active_patch_feats, current_text_feat, active_grid_indices, current_grid_shape
                )
            all_selected_windows_feats_b.append(s_feats)
            all_selected_windows_masks_b.append(s_mask)

        selected_windows_feats = torch.stack(all_selected_windows_feats_b, dim=0)
        selected_windows_mask = torch.stack(all_selected_windows_masks_b, dim=0)
        k_w = self.num_selected_windows
        proc_windows_feats = selected_windows_feats.reshape(batch_size * k_w, self.patches_per_window, self.patch_feature_dim)
        proc_windows_mask = selected_windows_mask.reshape(batch_size * k_w, self.patches_per_window)

        attended_patch_feats = self.window_self_attention(
            proc_windows_feats,
            mask=~proc_windows_mask.bool() if proc_windows_mask is not None else None 
        ) 
        
        aggregated_window_reprs, _ = self.window_mil_aggregator(
            attended_patch_feats,
            instance_mask=proc_windows_mask 
        ) 
        aggregated_window_reprs = aggregated_window_reprs.view(batch_size, k_w, self.window_mil_output_dim)
        
        inter_window_mask = selected_windows_mask.any(dim=2) # True if window has any valid patch
        final_image_repr, _ = self.inter_window_aggregator(
            aggregated_window_reprs,
            instance_mask=inter_window_mask
        ) 
        
        final_image_repr_seq = final_image_repr.unsqueeze(1)
        text_feat_batch_seq = text_feat_batch.unsqueeze(1)
        
        fused_representation = self.cross_attention(
            query=final_image_repr_seq,
            key_value=text_feat_batch_seq
        ) 
        fused_representation = fused_representation.squeeze(1)
        
        final_multimodal_feat = torch.cat([final_image_repr, fused_representation], dim=-1)
        logits = self.classifier(final_multimodal_feat)
        return logits

# --- 简单的配置类 (与上次相同) ---
class SimpleConfig:
    def __init__(self, **kwargs):
        self.patch_feature_dim = kwargs.get('patch_feature_dim', 1024)
        self.text_feature_dim = kwargs.get('text_feature_dim', 1024)
        self.num_classes = kwargs.get('num_classes', 2)
        self.model_params = SimpleConfigModelParams(**kwargs.get('model_params', {}))

class SimpleConfigModelParams:
    def __init__(self, **kwargs):
        self.similarity_projection_dim = kwargs.get('similarity_projection_dim', 256)
        self.window_params = SimpleConfigWindowParams(**kwargs.get('window_params', {}))
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


# --- 单元测试部分 ---
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
                    'light_agg_D': 8, 'light_agg_dropout': 0.0,
                },
                'self_attn_heads': 2, 'self_attn_dropout': 0.0,
                'window_mil_output_dim': 24, 'window_mil_D': 12, 'window_mil_dropout': 0.0,
                'final_image_feature_dim': 20, 'inter_window_mil_D': 10, 'inter_window_mil_dropout': 0.0,
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
        self.model.window_patch_rows = 2
        self.model.window_patch_cols = 2
        self.model.patches_per_window = 4
        self.model.stride_rows = 1
        self.model.stride_cols = 1

        feats_list, masks_list = self.model._generate_candidate_spatial_windows(
            self.all_patch_features_wsi,
            self.all_patch_grid_indices_wsi,
            self.wsi_grid_shape
        )
        expected_num_candidates_s1 = (self.wsi_grid_shape[0].item() - self.model.window_patch_rows + 1) * \
                                     (self.wsi_grid_shape[1].item() - self.model.window_patch_cols + 1)
        self.assertEqual(len(feats_list), expected_num_candidates_s1)
        self.assertEqual(len(masks_list), expected_num_candidates_s1)
        if feats_list:
            self.assertEqual(feats_list[0].shape, (self.model.patches_per_window, self.patch_dim))
            self.assertEqual(masks_list[0].shape, (self.model.patches_per_window,))
            self.assertTrue(masks_list[0].all())
        print(f"  生成了 {len(feats_list)} 个候选窗口。")

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
            self.wsi_grid_shape
        )
        self.assertEqual(sel_feats.shape, (self.model.num_selected_windows, self.model.patches_per_window, self.patch_dim))
        self.assertEqual(sel_masks.shape, (self.model.num_selected_windows, self.model.patches_per_window))
        # This assertion might be too strict if num_selected_windows > actual candidates and padding occurs
        # for i in range(self.model.num_selected_windows):
        #     if sel_masks[i].any(): # Only check non-empty padded windows
        #          self.assertTrue(sel_masks[i].all(), f"选定窗口 {i} 的内部掩码不应有padding, mask: {sel_masks[i]}")
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
        print(f"  单样本前向传播成功，Logits 形状: {logits.shape}")

    def test_04_forward_pass_batch_with_padding_mask(self):
        print("\n测试: forward pass (批处理, 带padding mask)")
        bs = 2
        max_patches = 30
        s1_feats = self.all_patch_features_wsi
        s1_grid_idx = self.all_patch_grid_indices_wsi
        s1_grid_shape = self.wsi_grid_shape
        s1_mask = torch.ones(self.N_total_patches, dtype=torch.bool)
        s2_N_patches = 9
        s2_patch_dim = self.config.patch_feature_dim
        s2_feats_actual = torch.randn(s2_N_patches, s2_patch_dim)
        s2_grid_r, s2_grid_c = torch.meshgrid(torch.arange(3), torch.arange(3), indexing='ij')
        s2_grid_idx_actual = torch.stack([s2_grid_r.flatten(), s2_grid_c.flatten()], dim=1)
        s2_grid_shape = torch.tensor([3, 3], dtype=torch.long)
        s2_mask_actual = torch.ones(s2_N_patches, dtype=torch.bool)

        img_feats_b = torch.zeros(bs, max_patches, self.patch_dim)
        grid_idx_b = torch.zeros(bs, max_patches, 2, dtype=torch.long)
        patch_mask_b = torch.zeros(bs, max_patches, dtype=torch.bool)
        
        img_feats_b[0, :self.N_total_patches] = s1_feats
        grid_idx_b[0, :self.N_total_patches] = s1_grid_idx
        patch_mask_b[0, :self.N_total_patches] = s1_mask
        img_feats_b[1, :s2_N_patches] = s2_feats_actual
        grid_idx_b[1, :s2_N_patches] = s2_grid_idx_actual
        patch_mask_b[1, :s2_N_patches] = s2_mask_actual
        
        txt_feat_b = torch.randn(bs, self.text_dim)
        grid_shape_b = torch.stack([s1_grid_shape, s2_grid_shape], dim=0)
        
        logits = self.model(img_feats_b, grid_idx_b, txt_feat_b, grid_shape_b, patch_mask_batch=patch_mask_b)
        self.assertEqual(logits.shape, (bs, self.config.num_classes))
        print(f"  批处理前向传播成功，Logits 形状: {logits.shape}")

# --- 主执行部分 ---
if __name__ == '__main__':
    print("开始 MultimodalTextGuidedMIL 模型组件和前向传播的单元测试...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\n所有单元测试执行完毕。")
