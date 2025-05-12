import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设这些是你 BMW/src/models/ 下已有的或将要完善的模块
# from .attention_layers import SelfAttention, CrossAttention # 确保参数匹配
# from .mil_aggregators import YourMILAggregator # 例如 AttentionMIL, MeanAggregator等

# 您可能需要从 BMW/src/models/attention_layers.py 导入或定义更具体的注意力实现
# 例如，SelfAttention 可能需要接收 (batch, num_patches_in_window, feature_dim)
# CrossAttention 可能需要接收 (batch, feature_dim1), (batch, feature_dim2)

# --- 为了演示，先定义一些占位符模块 ---
class PlaceholderSelfAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        print(f"PlaceholderSelfAttention initialized with dim {feature_dim}")
    def forward(self, x, mask=None): # x: (batch, seq_len, dim)
        x = self.norm(x)
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class PlaceholderCrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads=4, dropout=0.1):
        super().__init__()
        # 假设 K/V 来自同一源 (文本)，Q 来自图像
        # 如果需要，可以调整维度或添加投影层
        self.q_norm = nn.LayerNorm(query_dim)
        self.kv_norm = nn.LayerNorm(key_dim)
        # 确保 MultiheadAttention 的 embed_dim 与 q,k,v 的维度匹配
        # 这里假设 query_dim 就是 MultiheadAttention 的 embed_dim
        # 如果 key_dim 不同，通常 k 和 v 会被投影到 embed_dim
        self.proj_k = nn.Linear(key_dim, query_dim) if key_dim != query_dim else nn.Identity()
        self.proj_v = nn.Linear(key_dim, query_dim) if key_dim != query_dim else nn.Identity()
        self.attn = nn.MultiheadAttention(query_dim, num_heads, dropout=dropout, batch_first=True)
        print(f"PlaceholderCrossAttention initialized with q_dim {query_dim}, kv_dim {key_dim}")

    def forward(self, query, key_value, mask_kv=None): # query: (B, dim1), key_value: (B, dim2)
                                                  # 或者 query: (B, Nq, dim1), key_value: (B, Nkv, dim2)
        query_norm = self.q_norm(query)
        kv_norm = self.kv_norm(key_value)

        # 如果是单个向量，扩展为序列长度1
        if query_norm.ndim == 2:
            query_norm = query_norm.unsqueeze(1) # (B, 1, dim1)
        if kv_norm.ndim == 2:
            kv_norm = kv_norm.unsqueeze(1) # (B, 1, dim2)

        k = self.proj_k(kv_norm) # (B, Nkv, dim1)
        v = self.proj_v(kv_norm) # (B, Nkv, dim1)

        attn_output, _ = self.attn(query_norm, k, v, key_padding_mask=mask_kv) # attn_output: (B, Nq, dim1)
        return attn_output.squeeze(1) if query.ndim == 2 else attn_output


class PlaceholderMILAggregator(nn.Module):
    def __init__(self, feature_dim, L=512, D=128, K=1, dropout=0.1, mil_type='attention'):
        super().__init__()
        self.feature_dim = feature_dim
        self.mil_type = mil_type
        if mil_type == 'attention':
            self.attention_V = nn.Sequential(
                nn.Linear(self.feature_dim, D),
                nn.Tanh()
            )
            self.attention_U = nn.Sequential(
                nn.Linear(self.feature_dim, D),
                nn.Sigmoid()
            )
            self.attention_weights = nn.Linear(D, K)
        elif mil_type == 'mean':
            pass # 不需要额外层
        elif mil_type == 'max':
            pass # 不需要额外层
        else:
            raise NotImplementedError
        print(f"PlaceholderMILAggregator ({mil_type}) initialized with dim {feature_dim}")

    def forward(self, x, mask=None): # x: (batch, num_elements, dim)
        if self.mil_type == 'attention':
            # x might be (batch_size * num_windows, window_size, dim)
            # or (batch_size, num_windows, dim)
            A_V = self.attention_V(x)  # (batch, num_elements, D)
            A_U = self.attention_U(x)  # (batch, num_elements, D)
            A = self.attention_weights(A_V * A_U) # (batch, num_elements, K)
            A = F.softmax(A, dim=1)  # softmax over num_elements

            if mask is not None:
                # mask: (batch, num_elements) bool, True for valid, False for padding
                # expand mask to match A's shape (batch, num_elements, K)
                mask = mask.unsqueeze(-1).expand_as(A)
                A[~mask] = 0 # Zero out weights for padded elements
                A = A / (A.sum(dim=1, keepdim=True) + 1e-8) # Re-normalize

            M = torch.bmm(A.transpose(1, 2), x) # (batch, K, dim)
            return M.squeeze(1) # (batch, dim) if K=1
        elif self.mil_type == 'mean':
            if mask is not None:
                num_valid = mask.sum(dim=1, keepdim=True).float() # (batch, 1)
                x_masked = x * mask.unsqueeze(-1).float() # (batch, num_elements, dim)
                return x_masked.sum(dim=1) / (num_valid + 1e-8) # (batch, dim)
            else:
                return x.mean(dim=1) # (batch, dim)
        elif self.mil_type == 'max':
            if mask is not None:
                # Fill padded elements with a very small number before max pooling
                x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                return x_masked.max(dim=1)[0] # (batch, dim)
            else:
                return x.max(dim=1)[0] # (batch, dim)

# --- 主模型类 ---
class MultimodalTextGuidedMIL(nn.Module):
    def __init__(self, config): # config 是一个包含所有超参数的字典或对象
        super().__init__()
        self.config = config
        self.patch_feature_dim = config.get('patch_feature_dim', 1024) # MUSK patch 特征维度
        self.text_feature_dim = config.get('text_feature_dim', 768)   # MUSK text 特征维度
        self.num_classes = config.get('num_classes', 2)

        # 相似度计算相关参数
        self.similarity_projection_dim = config.get('similarity_projection_dim', 256)
        self.num_selected_windows = config.get('num_selected_windows', 5) # K_w, 选择多少个窗口
        self.window_size_patches = config.get('window_size_patches', 10) # 每个窗口包含多少 patch

        # 注意力机制和聚合器参数
        self.attn_heads = config.get('attn_heads', 4)
        self.attn_dropout = config.get('attn_dropout', 0.1)
        
        # 内部模块实例化
        # 步骤 2: 文本引导的 Patch 窗口选择 (这部分逻辑比较复杂，可能主要在 forward 中实现)
        # 可以选择性地为文本和图像 patch 特征添加投影层，以便在共同空间计算相似度
        self.text_projection_for_similarity = nn.Linear(self.text_feature_dim, self.similarity_projection_dim)
        self.patch_projection_for_similarity = nn.Linear(self.patch_feature_dim, self.similarity_projection_dim)

        # 步骤 3: 窗口内自注意力
        # 注意：这里的 feature_dim 应该是 patch_feature_dim
        self.window_self_attention = PlaceholderSelfAttention(
            feature_dim=self.patch_feature_dim,
            num_heads=self.attn_heads,
            dropout=self.attn_dropout
        )

        # 步骤 4: 窗口 MIL 聚合
        # 输入是 patch_feature_dim，输出也是 patch_feature_dim (如果MIL不改变维度)
        self.window_mil_aggregator = PlaceholderMILAggregator(
            feature_dim=self.patch_feature_dim,
            mil_type=config.get('window_mil_type', 'attention'), # 'attention', 'mean', 'max'
            L=config.get('window_mil_L', 256), # MIL 内部维度 (如果适用)
            D=config.get('window_mil_D', 128), # MIL 内部维度 (如果适用)
            dropout=config.get('window_mil_dropout', 0.1)
        )
        
        # 步骤 4.5 (可选): 如果选择了多个窗口，可能需要再次聚合这些窗口的表示
        # 这个聚合器的输入维度是上一步 MIL 的输出维度 (这里假设是 patch_feature_dim)
        self.inter_window_aggregator_dim_in = self.patch_feature_dim
        self.inter_window_aggregator_dim_out = config.get('final_image_feature_dim', 512) # 最终图片表征维度
        self.inter_window_aggregator = PlaceholderMILAggregator(
            feature_dim=self.inter_window_aggregator_dim_in,
            mil_type=config.get('inter_window_mil_type', 'attention'),
            # 可以让输出维度不同
            # 如果 PlaceholderMILAggregator 不直接支持输出维度变化，
            # 可能需要在其后加一个 nn.Linear(self.inter_window_aggregator_dim_in, self.inter_window_aggregator_dim_out)
        )
        # 临时加一个线性层来确保维度匹配，如果聚合器不改变维度
        if self.inter_window_aggregator_dim_in != self.inter_window_aggregator_dim_out:
            self.inter_window_projection = nn.Linear(self.inter_window_aggregator_dim_in, self.inter_window_aggregator_dim_out)
        else:
            self.inter_window_projection = nn.Identity()


        # 步骤 5: 图文交叉注意力
        # Query 是聚合后的图像特征，Key/Value 是文本特征
        self.cross_attention = PlaceholderCrossAttention(
            query_dim=self.inter_window_aggregator_dim_out, # 来自聚合后的图像窗口
            key_dim=self.text_feature_dim,       # 来自文本
            num_heads=self.attn_heads,
            dropout=self.attn_dropout
        )
        # 交叉注意力的输出维度通常与 query_dim 相同
        self.cross_attention_output_dim = self.inter_window_aggregator_dim_out

        # 步骤 6: 残差融合与拼接
        # "图片特征用残差进行和融合特征拼接"
        # 这里的 "图片特征" 指的是 inter_window_aggregator 的输出 (final_image_representation)
        # "融合特征" 指的是 cross_attention 的输出
        # 拼接后的维度是 final_image_feature_dim + cross_attention_output_dim
        self.fused_feature_dim = self.inter_window_aggregator_dim_out + self.cross_attention_output_dim
        
        # 步骤 7: 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_feature_dim, config.get('classifier_hidden_dim', 256)),
            nn.ReLU(),
            nn.Dropout(config.get('classifier_dropout', 0.25)),
            nn.Linear(config.get('classifier_hidden_dim', 256), self.num_classes)
        )

        print("MultimodalTextGuidedMIL model initialized.")

    def _select_text_guided_windows(self, image_patch_feats, text_feat, patch_coordinates=None):
        """
        实现步骤 2: 文本引导的 Patch 窗口选择
        输入:
            image_patch_feats: (batch_size, num_patches, patch_feature_dim)
            text_feat: (batch_size, text_feature_dim)
            patch_coordinates: (batch_size, num_patches, 2) [可选]
        输出:
            selected_windows_feats: (batch_size, num_selected_windows, window_size_patches, patch_feature_dim)
            selected_windows_mask: (batch_size, num_selected_windows, window_size_patches) [可选, bool, True for valid]
        """
        batch_size, num_patches, _ = image_patch_feats.shape

        # 1. 投影到共同空间 (可选，但推荐)
        proj_text_feat = self.text_projection_for_similarity(text_feat)  # (B, sim_dim)
        proj_patch_feats = self.patch_projection_for_similarity(image_patch_feats) # (B, N_p, sim_dim)

        # 2. 计算相似度 (以余弦相似度为例)
        #    proj_text_feat 扩展为 (B, 1, sim_dim) 以便广播
        similarity_scores = F.cosine_similarity(proj_text_feat.unsqueeze(1), proj_patch_feats, dim=2) # (B, N_p)

        # 3. 选择 Top-K patches 作为窗口的“锚点”或直接形成窗口
        #    这是一个简化实现：选择与文本最相似的 K_w * window_size 个 patches，然后重塑。
        #    更复杂的逻辑可能基于 patch_coordinates 进行空间窗口构建。
        
        # 这里简化为：基于最高相似度选择不重叠的窗口的起始点
        # 这种方法可能不是最优的，需要仔细设计
        
        # 先选择 num_selected_windows * window_size_patches 个最相似的 patches
        # (这种方式不保证窗口的连续性，仅为示例)
        # _, top_k_indices = torch.topk(similarity_scores, 
        #                               k=self.num_selected_windows * self.window_size_patches, 
        #                               dim=1) # (B, K_w * W_s)

        # 更合理的简化：选择 K_w 个“锚点”，然后从锚点扩展形成窗口
        # 如果 patch_coordinates 可用，可以实现滑动窗口或基于锚点的空间窗口
        
        # 示例：选择 K_w 个最高相似度的 patch 作为锚点
        _, anchor_indices = torch.topk(similarity_scores, k=self.num_selected_windows, dim=1) # (B, K_w)

        selected_windows_list = []
        # 假设我们简单地从每个锚点开始取 window_size_patches 个连续的 patch
        # 注意：这需要确保索引不越界，且 patch 在原始序列中是“连续”的才有意义
        # 实际应用中，这里的窗口形成逻辑会更复杂，可能需要padding和mask
        
        # 创建一个全零的tensor用于存储窗口特征，和一个mask
        selected_windows_feats_tensor = torch.zeros(
            batch_size, self.num_selected_windows, self.window_size_patches, self.patch_feature_dim,
            device=image_patch_feats.device, dtype=image_patch_feats.dtype
        )
        selected_windows_mask_tensor = torch.zeros(
            batch_size, self.num_selected_windows, self.window_size_patches,
            device=image_patch_feats.device, dtype=torch.bool
        )

        for i in range(batch_size):
            for j in range(self.num_selected_windows):
                anchor_idx = anchor_indices[i, j]
                # 简单地从锚点开始取，如果 patch_coordinates 没有使用
                start_idx = anchor_idx 
                end_idx = start_idx + self.window_size_patches
                
                # 边界处理和实际取的patch数量
                actual_end_idx = min(end_idx, num_patches)
                actual_len = actual_end_idx - start_idx
                
                if actual_len > 0:
                    selected_windows_feats_tensor[i, j, :actual_len, :] = \
                        image_patch_feats[i, start_idx:actual_end_idx, :]
                    selected_windows_mask_tensor[i, j, :actual_len] = True
        
        return selected_windows_feats_tensor, selected_windows_mask_tensor


    def forward(self, image_patch_feats, text_feat, patch_coordinates=None, patch_mask=None):
        """
        输入:
            image_patch_feats: (batch_size, num_total_patches, patch_feature_dim)
            text_feat: (batch_size, text_feature_dim) - e.g., "小叶炎症" 的 embedding
            patch_coordinates: (batch_size, num_total_patches, 2) [可选] - patch 坐标
            patch_mask: (batch_size, num_total_patches) [可选, bool] - True for valid patches
        """
        batch_size = image_patch_feats.shape[0]

        # --- 步骤 2: 文本引导的 Patch 窗口选择 ---
        # selected_windows_feats: (B, K_w, W_s, patch_dim) K_w=num_selected_windows, W_s=window_size_patches
        # selected_windows_mask: (B, K_w, W_s)
        selected_windows_feats, selected_windows_mask = self._select_text_guided_windows(
            image_patch_feats, text_feat, patch_coordinates
        )
        
        # 为了处理可变长度和后续的批处理注意力/聚合，可能需要 reshape 和处理 mask
        # (B, K_w, W_s, patch_dim) -> (B * K_w, W_s, patch_dim)
        k_w = self.num_selected_windows
        w_s = self.window_size_patches
        
        # Reshape for batch processing of windows
        # Each window is treated as an element in a new batch dimension
        proc_windows_feats = selected_windows_feats.view(batch_size * k_w, w_s, self.patch_feature_dim)
        proc_windows_mask = selected_windows_mask.view(batch_size * k_w, w_s) # True for valid patches

        # --- 步骤 3: 窗口内自注意力 ---
        # 输入: (B * K_w, W_s, patch_dim)
        # 输出: (B * K_w, W_s, patch_dim)
        # 注意：MultiheadAttention 的 key_padding_mask 是 True 表示忽略，所以用 ~proc_windows_mask
        attended_patch_feats_in_windows = self.window_self_attention(
            proc_windows_feats,
            mask=~proc_windows_mask if proc_windows_mask is not None else None
        )

        # --- 步骤 4: 窗口 MIL 聚合 ---
        # 将每个窗口的 W_s 个 patch 特征聚合成一个窗口表示
        # 输入: (B * K_w, W_s, patch_dim), mask: (B * K_w, W_s)
        # 输出: (B * K_w, patch_dim) [如果MIL聚合器输出维度与输入一致且K=1]
        aggregated_window_reprs = self.window_mil_aggregator(
            attended_patch_feats_in_windows,
            mask=proc_windows_mask
        )
        
        # Reshape back: (B * K_w, patch_dim) -> (B, K_w, patch_dim)
        aggregated_window_reprs = aggregated_window_reprs.view(batch_size, k_w, -1)
        #
        # --- 步骤 4.5 (可选但根据您的描述是需要的): 聚合多个窗口的表示得到单一图片级特征 ---
        # 输入: (B, K_w, patch_dim_after_win_mil)
        # 输出: (B, final_image_feature_dim)
        # 这里的 mask 应该是表示哪些窗口是有效的 (如果某些情况下 K_w 不是固定的，或者某些窗口是padding)
        # 假设所有 K_w 个窗口都是有效选出的
        inter_window_mask = torch.ones(batch_size, k_w, device=aggregated_window_reprs.device, dtype=torch.bool)

        final_image_representation_before_proj = self.inter_window_aggregator(
            aggregated_window_reprs,
            mask=inter_window_mask
        )
        final_image_representation = self.inter_window_projection(final_image_representation_before_proj) # (B, final_image_feature_dim)


        # --- 步骤 5: 图文交叉注意力 ---
        # Query: final_image_representation (B, final_image_feature_dim)
        # Key/Value: text_feat (B, text_feature_dim)
        # 输出: (B, cross_attention_output_dim)
        # 注意：cross_attention 的 key_padding_mask (如果文本序列有padding的话)
        # 这里假设 text_feat 是单个向量，没有序列padding
        fused_representation = self.cross_attention(
            query=final_image_representation, # (B, final_img_dim)
            key_value=text_feat               # (B, text_dim)
        )

        # --- 步骤 6: 残差融合与拼接 ---
        # "图片特征用残差进行和融合特征拼接"
        # 图片特征: final_image_representation (B, final_image_feature_dim)
        # 融合特征: fused_representation (B, cross_attention_output_dim)
        # 您的描述是 "图片特征用残差进行和融合特征拼接"
        # 可能是指：img_residual_component = final_image_representation + (部分融合特征，如图像对其的部分)
        # 然后 result = cat(img_residual_component, (另一部分融合特征，如文本对其的部分))
        #
        # 一种直接的理解是:
        # 1. final_image_representation 作为一路
        # 2. fused_representation (已经是图文融合结果) 作为另一路
        # 然后拼接:
        final_multimodal_feat = torch.cat([final_image_representation, fused_representation], dim=-1)
        # 如果"残差"指的是 final_image_representation 直接加入到最终的分类器输入中，
        # 并且也参与了 fused_representation 的计算（通过 cross_attention），
        # 那么这里的拼接已经包含了 final_image_representation。

        # --- 步骤 7: 分类 ---
        # 输入: (B, self.fused_feature_dim)
        logits = self.classifier(final_multimodal_feat) # (B, num_classes)

        return logits

# --- 使用示例 (需要配合配置文件) ---
if __name__ == '__main__':
    # 这是一个非常简化的配置示例
    dummy_config = {
        'patch_feature_dim': 1024,
        'text_feature_dim': 768,
        'num_classes': 3,
        'similarity_projection_dim': 256,
        'num_selected_windows': 5,
        'window_size_patches': 8, # 8 patches per window
        'attn_heads': 4,
        'attn_dropout': 0.1,
        'window_mil_type': 'attention', # MIL for patches within a window
        'window_mil_L': 128,
        'window_mil_D': 64,
        'inter_window_mil_type': 'attention', # MIL for aggregating window representations
        'final_image_feature_dim': 512, # Output dim of inter_window_aggregator path
        'classifier_hidden_dim': 256,
        'classifier_dropout': 0.2
    }

    class Config: # 简单模拟OmegaConf或类似的配置对象
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def get(self, key, default=None):
            return getattr(self, key, default)

    config_obj = Config(**dummy_config)

    # 实例化模型
    model = MultimodalTextGuidedMIL(config_obj)

    # 创建虚拟输入数据
    bs = 4
    num_total_patches = 100 # 每张 WSI 的总 patch 数
    
    dummy_image_patches = torch.randn(bs, num_total_patches, config_obj.patch_feature_dim)
    dummy_text_feature = torch.randn(bs, config_obj.text_feature_dim)
    # dummy_patch_coords = torch.rand(bs, num_total_patches, 2) * 224 # 假设patch大小224
    # dummy_patch_mask = torch.ones(bs, num_total_patches, dtype=torch.bool)
    # # 随机让一些patch无效 (示例)
    # if num_total_patches > 10:
    #     dummy_patch_mask[:, -10:] = False


    # 模型前向传播
    try:
        logits = model(dummy_image_patches, dummy_text_feature) #, patch_coordinates=dummy_patch_coords, patch_mask=dummy_patch_mask)
        print("Model forward pass successful!")
        print("Logits shape:", logits.shape) # 期望: (batch_size, num_classes)
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        import traceback
        traceback.print_exc()