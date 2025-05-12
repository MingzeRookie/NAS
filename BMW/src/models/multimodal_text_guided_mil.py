import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # 用于 _generate_candidate_spatial_windows 中的网格操作

# 假设您的自定义模块位于以下路径
# 您需要确保这些导入路径相对于您的项目结构是正确的
# 并且这些类 (SelfAttentionLayer, CrossAttentionLayer, AttentionMIL) 存在且接口兼容
try:
    from BMW.src.models.attention_layers import SelfAttentionLayer, CrossAttentionLayer
    from BMW.src.models.mil_aggregators import AttentionMIL # 或者您使用的其他聚合器类
except ImportError:
    print("警告: 未能从 BMW.src.models 导入自定义注意力或聚合器层。将使用占位符。")
    print("请确保 SelfAttentionLayer, CrossAttentionLayer, AttentionMIL 类存在且路径正确。")
    # --- 占位符模块 (如果导入失败) ---
    class SelfAttentionLayer(nn.Module): # 简化的自注意力占位符
        def __init__(self, feature_dim, num_heads=4, dropout=0.1):
            super().__init__()
            self.norm = nn.LayerNorm(feature_dim)
            self.attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
            print(f"占位符 SelfAttentionLayer 初始化，维度 {feature_dim}")
        def forward(self, x, mask=None): # x: (batch, seq_len, dim), mask: (batch, seq_len)
            x = self.norm(x)
            # MultiheadAttention 的 key_padding_mask 需要 True 表示忽略
            attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
            return attn_output

    class CrossAttentionLayer(nn.Module): # 简化的交叉注意力占位符
        def __init__(self, query_dim, key_dim, num_heads=4, dropout=0.1):
            super().__init__()
            self.q_norm = nn.LayerNorm(query_dim)
            self.kv_norm = nn.LayerNorm(key_dim)
            # MultiheadAttention 的 embed_dim 通常与 query_dim 相同
            # 如果 key_dim 不同，k 和 v 通常会被投影到 embed_dim
            self.proj_k = nn.Linear(key_dim, query_dim) if key_dim != query_dim else nn.Identity()
            self.proj_v = nn.Linear(key_dim, query_dim) if key_dim != query_dim else nn.Identity()
            self.attn = nn.MultiheadAttention(query_dim, num_heads, dropout=dropout, batch_first=True)
            print(f"占位符 CrossAttentionLayer 初始化，查询维度 {query_dim}, 键/值维度 {key_dim}")
        def forward(self, query, key_value, kv_mask=None): # query: (B, Nq, Dq), key_value: (B, Nkv, Dkv)
            query_norm = self.q_norm(query)
            kv_norm = self.kv_norm(key_value)
            k = self.proj_k(kv_norm)
            v = self.proj_v(kv_norm)
            # MultiheadAttention 的 key_padding_mask 需要 True 表示忽略
            attn_output, _ = self.attn(query_norm, k, v, key_padding_mask=kv_mask)
            return attn_output

    class AttentionMIL(nn.Module): # 简化的AttentionMIL占位符
        def __init__(self, input_dim, hidden_dim_att, output_dim_att=1, dropout_att=0.25, output_dim=None):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim if output_dim is not None else input_dim # 默认输出维度与输入一致

            self.attention_V = nn.Sequential(
                nn.Linear(input_dim, hidden_dim_att),
                nn.Tanh()
            )
            self.attention_U = nn.Sequential(
                nn.Linear(input_dim, hidden_dim_att),
                nn.Sigmoid()
            )
            self.attention_weights = nn.Linear(hidden_dim_att, output_dim_att) # K 通常为1
            
            if self.input_dim != self.output_dim and output_dim_att == 1: # 如果需要改变特征维度
                self.feature_transform = nn.Linear(self.input_dim, self.output_dim)
            else:
                self.feature_transform = nn.Identity()

            print(f"占位符 AttentionMIL 初始化，输入维度 {input_dim}, 输出维度 {self.output_dim}")

        def forward(self, x, instance_mask=None): # x: (B, N, D_in), instance_mask: (B, N)
            A_V = self.attention_V(x)  # (B, N, D_h)
            A_U = self.attention_U(x)  # (B, N, D_h)
            att_raw = self.attention_weights(A_V * A_U) # (B, N, K)
            
            if instance_mask is not None:
                # mask: True for valid, False for padding. We want to set weights of padding to -inf
                att_raw[~instance_mask.unsqueeze(-1).expand_as(att_raw)] = float('-inf')
            
            att_scores = F.softmax(att_raw, dim=1) # (B, N, K)
            
            # Weighted sum of features
            # x: (B, N, D_in), att_scores (transposed): (B, K, N)
            # M: (B, K, D_in)
            M = torch.bmm(att_scores.transpose(1, 2), x)
            
            if M.size(1) == 1: # 如果 K=1
                M = M.squeeze(1) # (B, D_in)
                M = self.feature_transform(M) # (B, D_out)
            else: # 如果 K > 1, 可能需要进一步处理或确保 feature_transform 能处理 (B, K, D_in)
                # 为简单起见，假设 K=1 时才进行维度变换
                if self.input_dim != self.output_dim:
                    # 这个变换可能需要更复杂的处理，如果 K > 1 且需要保持 K
                    print("警告: K > 1 时的特征变换未在占位符中明确处理。")
                M = self.feature_transform(M)


            return M, att_scores # 返回聚合后的特征和注意力分数


class MultimodalTextGuidedMIL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 从配置中提取参数
        self.patch_feature_dim = config.patch_feature_dim
        self.text_feature_dim = config.text_feature_dim
        self.num_classes = config.num_classes

        # 窗口选择参数
        self.window_patch_rows = config.model_params.window_params.patch_rows
        self.window_patch_cols = config.model_params.window_params.patch_cols
        self.patches_per_window = self.window_patch_rows * self.window_patch_cols
        self.stride_rows = config.model_params.window_params.stride_rows
        self.stride_cols = config.model_params.window_params.stride_cols
        self.num_selected_windows = config.model_params.window_params.num_selected_windows # K_w
        self.pre_similarity_window_agg_type = config.model_params.window_params.pre_similarity_window_agg_type

        # 相似度计算投影层
        self.similarity_projection_dim = config.model_params.similarity_projection_dim
        self.text_proj_sim = nn.Linear(self.text_feature_dim, self.similarity_projection_dim)
        self.patch_proj_sim = nn.Linear(self.patch_feature_dim, self.similarity_projection_dim)

        # 轻量级窗口聚合器 (用于计算与文本的相似度前)
        if self.pre_similarity_window_agg_type == 'attention_light':
            self.light_window_aggregator = AttentionMIL(
                input_dim=self.patch_feature_dim,
                hidden_dim_att=config.model_params.window_params.light_agg_D, # D
                # output_dim_att=1 (K) is default in AttentionMIL for single vector output
                dropout_att=config.model_params.window_params.light_agg_dropout,
                output_dim=self.patch_feature_dim # 保持维度以便后续投影
            )
        elif self.pre_similarity_window_agg_type == 'mean':
            self.light_window_aggregator = lambda x, mask: (x * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-6) if mask is not None else x.mean(dim=1)
        elif self.pre_similarity_window_agg_type == 'max':
             self.light_window_aggregator = lambda x, mask: x.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1)[0] if mask is not None else x.max(dim=1)[0]
        else:
            raise ValueError(f"不支持的 pre_similarity_window_agg_type: {self.pre_similarity_window_agg_type}")

        # 步骤 3: 窗口内自注意力
        self.window_self_attention = SelfAttentionLayer(
            feature_dim=self.patch_feature_dim,
            num_heads=config.model_params.self_attn_heads,
            dropout=config.model_params.self_attn_dropout
        )

        # 步骤 4: 窗口内 MIL 聚合 (聚合一个窗口内的 patches)
        self.window_mil_output_dim = config.model_params.window_mil_output_dim
        self.window_mil_aggregator = AttentionMIL(
            input_dim=self.patch_feature_dim, # 输入是自注意力后的 patch 特征
            hidden_dim_att=config.model_params.window_mil_D,
            dropout_att=config.model_params.window_mil_dropout,
            output_dim=self.window_mil_output_dim # 输出维度可配置
        )

        # 步骤 4.5: 窗口间 MIL 聚合 (聚合 K_w 个窗口的表示)
        self.final_image_feature_dim = config.model_params.final_image_feature_dim
        self.inter_window_aggregator = AttentionMIL(
            input_dim=self.window_mil_output_dim, # 输入是上一步聚合器的输出
            hidden_dim_att=config.model_params.inter_window_mil_D,
            dropout_att=config.model_params.inter_window_mil_dropout,
            output_dim=self.final_image_feature_dim # 最终图像分支的输出维度
        )

        # 步骤 5: 图文交叉注意力
        # Query: 图像特征 (final_image_feature_dim)
        # Key/Value: 文本特征 (text_feature_dim)
        # Output: 通常与 Query 维度相同
        self.cross_attention_output_dim = self.final_image_feature_dim
        self.cross_attention = CrossAttentionLayer(
            query_dim=self.final_image_feature_dim,
            key_dim=self.text_feature_dim,
            num_heads=config.model_params.cross_attn_heads,
            dropout=config.model_params.cross_attn_dropout
        )
        
        # 步骤 6: 残差融合与拼接
        # 拼接 final_image_representation 和 cross_attention 的输出
        self.fused_feature_dim = self.final_image_feature_dim + self.cross_attention_output_dim
        
        # 步骤 7: 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_feature_dim, config.model_params.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.model_params.classifier_dropout),
            nn.Linear(config.model_params.classifier_hidden_dim, self.num_classes)
        )
        print("MultimodalTextGuidedMIL 模型初始化完成。")

    def _generate_candidate_spatial_windows(self,
                                            all_patch_features_wsi, # [N_total, D_patch]
                                            all_patch_grid_indices_wsi, # [N_total, 2] (row, col)
                                            wsi_grid_shape): # [max_rows, max_cols]
        """
        为单个 WSI 生成所有候选的空间窗口及其 patch 特征和掩码。
        """
        num_total_patches, _ = all_patch_features_wsi.shape
        max_r, max_c = wsi_grid_shape[0].item(), wsi_grid_shape[1].item()

        candidate_windows_feats_list = []
        candidate_windows_masks_list = []

        # 创建一个稀疏网格查找表: (r, c) -> 原始 patch 索引
        coord_to_idx_map = {tuple(coord.tolist()): i for i, coord in enumerate(all_patch_grid_indices_wsi)}

        for r_start in range(0, max_r - self.window_patch_rows + 1, self.stride_rows):
            for c_start in range(0, max_c - self.window_patch_cols + 1, self.stride_cols):
                current_window_patch_indices = []
                for r_offset in range(self.window_patch_rows):
                    for c_offset in range(self.window_patch_cols):
                        # 当前 patch 在整个 WSI 网格中的绝对坐标
                        abs_r, abs_c = r_start + r_offset, c_start + c_offset
                        if (abs_r, abs_c) in coord_to_idx_map:
                            current_window_patch_indices.append(coord_to_idx_map[(abs_r, abs_c)])
                
                # 如果窗口内实际找到的 patch 数大于0 (避免空窗口)
                if len(current_window_patch_indices) > 0:
                    # 提取特征并创建掩码
                    window_feats = all_patch_features_wsi[current_window_patch_indices] # [actual_patches, D_patch]
                    
                    # Pad 到 self.patches_per_window
                    num_actual_patches = window_feats.shape[0]
                    
                    # 创建特征张量和掩码张量
                    padded_window_feats = torch.zeros(self.patches_per_window, self.patch_feature_dim,
                                                      device=all_patch_features_wsi.device, dtype=all_patch_features_wsi.dtype)
                    window_mask = torch.zeros(self.patches_per_window, device=all_patch_features_wsi.device, dtype=torch.bool)
                    
                    if num_actual_patches > 0:
                        # 如果实际 patch 数超过了窗口容量 (理论上不应发生，除非窗口选择逻辑有误或 patch 重叠严重)
                        # 这里简单截断，但更好的做法是确保窗口定义正确
                        num_to_fill = min(num_actual_patches, self.patches_per_window)
                        padded_window_feats[:num_to_fill] = window_feats[:num_to_fill]
                        window_mask[:num_to_fill] = True
                    
                    candidate_windows_feats_list.append(padded_window_feats) # [W*W, D_patch]
                    candidate_windows_masks_list.append(window_mask)         # [W*W]
        
        return candidate_windows_feats_list, candidate_windows_masks_list

    def _select_text_guided_windows(self,
                                    all_patch_features_wsi,    # [N_total, D_patch]
                                    text_feat_wsi,             # [D_text]
                                    all_patch_grid_indices_wsi,# [N_total, 2]
                                    wsi_grid_shape):           # [2] (max_r, max_c)
        """
        为单个 WSI 选择与文本最相关的 K_w 个空间窗口。
        """
        candidate_windows_feats_list, candidate_windows_masks_list = \
            self._generate_candidate_spatial_windows(all_patch_features_wsi, all_patch_grid_indices_wsi, wsi_grid_shape)

        if not candidate_windows_feats_list: # 如果没有生成任何候选窗口
            # 返回填充的零张量
            padded_selected_feats = torch.zeros(self.num_selected_windows, self.patches_per_window, self.patch_feature_dim,
                                                device=all_patch_features_wsi.device, dtype=all_patch_features_wsi.dtype)
            padded_selected_masks = torch.zeros(self.num_selected_windows, self.patches_per_window,
                                                device=all_patch_features_wsi.device, dtype=torch.bool)
            return padded_selected_feats, padded_selected_masks

        # 对每个候选窗口的 patch 特征进行初步聚合
        aggregated_candidate_reprs = []
        for i, window_feats in enumerate(candidate_windows_feats_list):
            # window_feats: [W*W, D_patch], window_mask: [W*W]
            window_mask = candidate_windows_masks_list[i]
            # unsqueeze(0) to make it [1, W*W, D_patch] for aggregator
            # light_window_aggregator 可能返回 (features, att_scores)
            agg_repr, _ = self.light_window_aggregator(window_feats.unsqueeze(0), instance_mask=window_mask.unsqueeze(0)) if self.pre_similarity_window_agg_type == 'attention_light' else (self.light_window_aggregator(window_feats.unsqueeze(0), window_mask.unsqueeze(0)), None)
            aggregated_candidate_reprs.append(agg_repr.squeeze(0)) # [D_patch]

        stacked_candidate_reprs = torch.stack(aggregated_candidate_reprs) # [Num_candidates, D_patch]

        # 计算与文本特征的相似度
        proj_text_feat = self.text_proj_sim(text_feat_wsi) # [D_sim]
        proj_candidate_reprs = self.patch_proj_sim(stacked_candidate_reprs) # [Num_candidates, D_sim]
        
        similarity_scores = F.cosine_similarity(proj_text_feat.unsqueeze(0), proj_candidate_reprs, dim=1) # [Num_candidates]

        # 选择 Top-K_w 个窗口
        num_actual_candidates = similarity_scores.shape[0]
        num_to_select = min(self.num_selected_windows, num_actual_candidates)

        final_selected_feats_list = []
        final_selected_masks_list = []

        if num_to_select > 0:
            _, top_k_indices = torch.topk(similarity_scores, k=num_to_select, dim=0) # [k_actual]
            for idx in top_k_indices:
                final_selected_feats_list.append(candidate_windows_feats_list[idx])
                final_selected_masks_list.append(candidate_windows_masks_list[idx])
        
        # Pad 到 K_w 个窗口
        # 使用列表推导和 torch.stack 来构建初始张量（如果列表非空）
        if final_selected_feats_list:
            padded_selected_feats = torch.stack(final_selected_feats_list, dim=0) # [k_actual, W*W, D_patch]
            padded_selected_masks = torch.stack(final_selected_masks_list, dim=0) # [k_actual, W*W]
        else: # 如果一个窗口都选不出来
            padded_selected_feats = torch.empty(0, self.patches_per_window, self.patch_feature_dim, device=all_patch_features_wsi.device, dtype=all_patch_features_wsi.dtype)
            padded_selected_masks = torch.empty(0, self.patches_per_window, device=all_patch_features_wsi.device, dtype=torch.bool)


        num_padding_windows = self.num_selected_windows - num_to_select
        if num_padding_windows > 0:
            padding_f = torch.zeros(num_padding_windows, self.patches_per_window, self.patch_feature_dim,
                                    device=all_patch_features_wsi.device, dtype=all_patch_features_wsi.dtype)
            padding_m = torch.zeros(num_padding_windows, self.patches_per_window,
                                    device=all_patch_features_wsi.device, dtype=torch.bool) # False for padded items
            
            padded_selected_feats = torch.cat([padded_selected_feats, padding_f], dim=0)
            padded_selected_masks = torch.cat([padded_selected_masks, padding_m], dim=0)
            
        return padded_selected_feats, padded_selected_masks # [K_w, W*W, D_patch], [K_w, W*W]

    def forward(self, image_patch_features_batch, patch_grid_indices_batch,
                  text_feat_batch, grid_shapes_batch, original_patch_coordinates_batch=None, # original_patch_coordinates_batch 未在此实现中使用
                  patch_mask_batch=None): # patch_mask_batch (B, N_total) 指示哪些原始patch有效
        """
        模型的前向传播。
        输入:
            image_patch_features_batch (Tensor): [B, N_max_patches, D_patch] (可能已padding到N_max_patches)
            patch_grid_indices_batch (Tensor): [B, N_max_patches, 2] (row, col indices)
            text_feat_batch (Tensor): [B, D_text]
            grid_shapes_batch (Tensor): [B, 2] (max_r, max_c for each WSI)
            patch_mask_batch (Tensor, optional): [B, N_max_patches] bool, True for valid patches.
        """
        batch_size = image_patch_features_batch.shape[0]
        
        all_selected_windows_feats_b = []
        all_selected_windows_masks_b = []

        for i in range(batch_size):
            # 提取单个 WSI 的数据
            current_patch_feats = image_patch_features_batch[i]    # [N_max, D_patch]
            current_grid_indices = patch_grid_indices_batch[i] # [N_max, 2]
            current_text_feat = text_feat_batch[i]             # [D_text]
            current_grid_shape = grid_shapes_batch[i]          # [2]
            
            # 如果有 patch_mask_batch，则只使用有效的 patches 和它们的坐标
            if patch_mask_batch is not None and patch_mask_batch[i] is not None:
                valid_patches_mask_i = patch_mask_batch[i] # [N_max]
                active_patch_feats = current_patch_feats[valid_patches_mask_i]    # [N_active, D_patch]
                active_grid_indices = current_grid_indices[valid_patches_mask_i] # [N_active, 2]
            else: # 假设所有提供的 patch 都有效
                active_patch_feats = current_patch_feats
                active_grid_indices = current_grid_indices

            if active_patch_feats.shape[0] == 0: # 如果这个WSI没有有效的patch
                 # 返回填充的零张量，与 _select_text_guided_windows 的无候选窗口情况类似
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

        # 堆叠批次结果
        # selected_windows_feats: [B, K_w, W*W, D_patch]
        # selected_windows_mask:  [B, K_w, W*W] (True for valid patch in window)
        selected_windows_feats = torch.stack(all_selected_windows_feats_b, dim=0)
        selected_windows_mask = torch.stack(all_selected_windows_masks_b, dim=0)

        # --- 后续步骤 ---
        k_w = self.num_selected_windows
        
        # Reshape for batch processing of windows: [B * K_w, W*W, D_patch]
        proc_windows_feats = selected_windows_feats.reshape(batch_size * k_w, self.patches_per_window, self.patch_feature_dim)
        proc_windows_mask = selected_windows_mask.reshape(batch_size * k_w, self.patches_per_window)

        # 步骤 3: 窗口内自注意力
        # self_attention mask: True 表示忽略
        attended_patch_feats = self.window_self_attention(
            proc_windows_feats,
            mask=~proc_windows_mask if proc_windows_mask is not None else None
        ) # [B * K_w, W*W, D_patch]

        # 步骤 4: 窗口内 MIL 聚合
        # mil_aggregator mask: True 表示有效
        aggregated_window_reprs, _ = self.window_mil_aggregator(
            attended_patch_feats,
            instance_mask=proc_windows_mask
        ) # [B * K_w, window_mil_output_dim]
        
        # Reshape back: [B, K_w, window_mil_output_dim]
        aggregated_window_reprs = aggregated_window_reprs.view(batch_size, k_w, self.window_mil_output_dim)
        
        # 步骤 4.5: 窗口间 MIL 聚合
        # 创建 inter_window_mask: [B, K_w], True 如果窗口至少有一个有效patch
        inter_window_mask = selected_windows_mask.any(dim=2) # 如果一个窗口全是padding, mask会是False

        final_image_repr, _ = self.inter_window_aggregator(
            aggregated_window_reprs,
            instance_mask=inter_window_mask
        ) # [B, final_image_feature_dim]

        # 步骤 5: 图文交叉注意力
        # query: [B, 1, D_img_final], key_value: [B, 1, D_text] (如果文本是单个向量)
        # 如果 CrossAttentionLayer 期望序列输入，需要 unsqueeze
        final_image_repr_seq = final_image_repr.unsqueeze(1) # [B, 1, D_final_img]
        text_feat_batch_seq = text_feat_batch.unsqueeze(1)   # [B, 1, D_text]

        fused_representation = self.cross_attention(
            query=final_image_repr_seq,
            key_value=text_feat_batch_seq
        ).squeeze(1) # [B, cross_attention_output_dim], cross_attn_out_dim == final_img_dim

        # 步骤 6: 残差融合与拼接
        # 拼接 final_image_representation 和 cross_attention 的输出
        final_multimodal_feat = torch.cat([final_image_repr, fused_representation], dim=-1)
        
        # 步骤 7: 分类器
        logits = self.classifier(final_multimodal_feat) # [B, num_classes]

        return logits


# --- 简单的配置类和使用示例 ---
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
        self.window_mil_D = kwargs.get('window_mil_D', 128) # hidden_dim_att for window_mil
        self.window_mil_dropout = kwargs.get('window_mil_dropout', 0.1)
        
        self.final_image_feature_dim = kwargs.get('final_image_feature_dim', 512)
        self.inter_window_mil_D = kwargs.get('inter_window_mil_D', 64) # hidden_dim_att for inter_window_mil
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
        self.light_agg_D = kwargs.get('light_agg_D', 64) # hidden_dim_att for light_agg
        self.light_agg_dropout = kwargs.get('light_agg_dropout', 0.1)


if __name__ == '__main__':
    print("开始 MultimodalTextGuidedMIL 模型测试...")
    
    # 定义一个示例配置 (通常这些会从 YAML 文件加载)
    dummy_config_dict = {
        'patch_feature_dim': 1024,
        'text_feature_dim': 1024,
        'num_classes': 2,
        'model_params': {
            'similarity_projection_dim': 128,
            'window_params': {
                'patch_rows': 3,
                'patch_cols': 3,
                'stride_rows': 2, # 允许重叠
                'stride_cols': 2, # 允许重叠
                'num_selected_windows': 4,
                'pre_similarity_window_agg_type': 'attention_light', # 'mean', 'max', 'attention_light'
                'light_agg_D': 64,
                'light_agg_dropout': 0.1,
            },
            'self_attn_heads': 4,
            'self_attn_dropout': 0.1,
            'window_mil_output_dim': 256, # 输出维度
            'window_mil_D': 128, 
            'window_mil_dropout': 0.1,
            'final_image_feature_dim': 256, # 最终图像分支输出
            'inter_window_mil_D': 64,
            'inter_window_mil_dropout': 0.1,
            'cross_attn_heads': 4,
            'cross_attn_dropout': 0.1,
            'classifier_hidden_dim': 128,
            'classifier_dropout': 0.2,
        }
    }
    config = SimpleConfig(**dummy_config_dict)

    # 实例化模型
    model = MultimodalTextGuidedMIL(config)
    # print(model)

    # 创建虚拟输入数据
    batch_size = 2
    max_patches_per_wsi = 200 # 假设数据加载器已将每个WSI的patch数padding到此值
    
    # (B, N_max_patches, D_patch)
    image_feats_batch = torch.randn(batch_size, max_patches_per_wsi, config.patch_feature_dim)
    # (B, N_max_patches, 2) - 假设这些是网格索引 (r,c)
    patch_grid_indices_batch = torch.randint(0, 30, (batch_size, max_patches_per_wsi, 2), dtype=torch.long)
    # (B, D_text)
    text_feat_batch = torch.randn(batch_size, config.text_feature_dim)
    # (B, 2) - 每个WSI的网格形状 (max_r, max_c)
    grid_shapes_batch = torch.tensor([[25, 28], [28, 30]], dtype=torch.long) 
    
    # (B, N_max_patches) - True表示有效patch
    patch_mask_batch = torch.ones(batch_size, max_patches_per_wsi, dtype=torch.bool)
    # 模拟某些WSI的patch较少
    patch_mask_batch[0, 150:] = False 
    patch_mask_batch[1, 180:] = False

    print(f"\n输入 image_feats_batch 形状: {image_feats_batch.shape}")
    print(f"输入 patch_grid_indices_batch 形状: {patch_grid_indices_batch.shape}")
    print(f"输入 text_feat_batch 形状: {text_feat_batch.shape}")
    print(f"输入 grid_shapes_batch 形状: {grid_shapes_batch.shape}")
    print(f"输入 patch_mask_batch 形状: {patch_mask_batch.shape}")

    # 模型前向传播
    try:
        print("\n执行模型前向传播...")
        logits = model(image_feats_batch, patch_grid_indices_batch, text_feat_batch, grid_shapes_batch, patch_mask_batch=patch_mask_batch)
        print("模型前向传播成功!")
        print("Logits 形状:", logits.shape) # 期望: (batch_size, num_classes)
        assert logits.shape == (batch_size, config.num_classes)
    except Exception as e:
        print(f"模型前向传播时发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nMultimodalTextGuidedMIL 模型测试结束。")
