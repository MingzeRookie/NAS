import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 设备选择 ---
if torch.cuda.is_available():
    if torch.cuda.device_count() > 1: # 确保至少有2个GPU (0 and 1)
        device = torch.device("cuda:1")
        print(f"将使用 cuda:1 设备。")
    else:
        device = torch.device("cuda:0")
        print(f"警告: cuda:1 不可用 (总共 {torch.cuda.device_count()} 个GPU)。将尝试使用 cuda:0。")
else:
    device = torch.device("cpu")
    print(f"警告: CUDA 不可用。将使用 CPU。")

# 设定随机种子以便复现
torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


# --- 0. 辅助模块定义 (与之前相同) ---
class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        # MultiheadAttention expects query, key, value
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + self.dropout(attn_output) # Residual connection
        x = self.norm(x)
        return x

class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1024, dropout_rate=0.25):
        super(AttentionMIL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Attention network
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

        # Bottleneck layer for final feature
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )


    def forward(self, bag_feats):
        # bag_feats: (N, D) where N is number of instances in the bag, D is feature dimension
        # Or (1, N, D) if batch_size is 1
        if bag_feats.dim() == 2:
            bag_feats = bag_feats.unsqueeze(0) # Add batch dimension: (1, N, D)

        # Calculate attention scores
        A_V = self.attention_V(bag_feats)  # (1, N, H)
        A_U = self.attention_U(bag_feats)  # (1, N, H)
        A = self.attention_weights(A_V * A_U) # (1, N, 1) element-wise multiplication
        A = torch.transpose(A, 2, 1)  # (1, 1, N)
        A = F.softmax(A, dim=2)  # Softmax over instances in the bag

        # Weighted sum of features
        M = torch.bmm(A, bag_feats).squeeze(1)  # (1, 1, N) bmm (1, N, D) -> (1, 1, D) -> (1,D)
        
        # Pass through bottleneck layer
        final_feature = self.bottleneck(M) # (1, output_dim)
        return final_feature, A.squeeze(0) # Return feature and attention weights


# --- 1. 数据加载 ---
print("--- 1. 数据加载 ---")
num_patches = 500 # 示例 patch 数量
feature_dim = 1024

# 模拟图像 patch 特征数据加载
dummy_image_data_path = '/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/MUSK-feature/1819360.pt'
dummy_image_data = {
    'bag_feats': torch.randn(num_patches, feature_dim, dtype=torch.float16),
    'coords': torch.stack([
        torch.repeat_interleave(torch.arange(0, int(np.sqrt(num_patches)) * 10, 10), int(np.sqrt(num_patches))),
        torch.tile(torch.arange(0, int(np.sqrt(num_patches)) * 10, 10), (int(np.sqrt(num_patches)),))
    ], dim=1).int()[:num_patches]
}
# torch.save(dummy_image_data, dummy_image_data_path) # 取消注释以创建虚拟文件

# 模拟文本特征数据加载
dummy_text_data_path = '/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/MUSK-feature/MUSK-text-feature/averaged_text_feature.pt'
dummy_text_feature_tensor = torch.randn(1, feature_dim, dtype=torch.float16)
# torch.save(dummy_text_feature_tensor, dummy_text_data_path) # 取消注释以创建虚拟文件

# 实际使用时请替换为您的文件名
# image_data_path = '1819260.pt'
# text_feature_path = 'your_text_feature.pt'

# 加载图像特征数据
# image_data = torch.load(image_data_path, map_location='cpu') # 先加载到CPU，再转移
image_data = dummy_image_data # 使用虚拟数据
bag_feats = image_data['bag_feats'].float().to(device) # 转换为 float32 并移动到设备
coords = image_data['coords'].to(device) # 移动坐标到设备
print(f"加载的 patch 特征形状: {bag_feats.shape}, dtype: {bag_feats.dtype}, device: {bag_feats.device}")
print(f"加载的坐标形状: {coords.shape}, device: {coords.device}")

# 加载文本特征数据
# text_feat_tensor = torch.load(text_feature_path, map_location='cpu') # 先加载到CPU
text_feat_tensor = dummy_text_feature_tensor # 使用虚拟数据
text_feat = text_feat_tensor.float().to(device) # 转换为 float32 并移动到设备
print(f"加载的文本特征形状: {text_feat.shape}, dtype: {text_feat.dtype}, device: {text_feat.device}\n")


# --- 2. 初始余弦相似度计算 ---
print("--- 2. 初始余弦相似度计算 ---")
cos_similarities = F.cosine_similarity(bag_feats, text_feat.squeeze(0), dim=1)
print(f"计算得到的余弦相似度形状: {cos_similarities.shape}, device: {cos_similarities.device}")
# .tolist() or .cpu().tolist() for cleaner print if tensor is on GPU
print(f"前5个余弦相似度: {cos_similarities[:5].cpu().tolist()}\n")


# --- 3. 最高相似度 Patch 识别 ---
print("--- 3. 最高相似度 Patch 识别 ---")
max_sim_score, max_sim_idx = torch.max(cos_similarities, dim=0)
# highest_sim_patch_feat = bag_feats[max_sim_idx] # This will be on GPU
# highest_sim_patch_coord = coords[max_sim_idx] # This will be on GPU
print(f"最高相似度得分: {max_sim_score.item()}") # .item() implicitly moves to CPU
print(f"最高相似度 Patch 索引: {max_sim_idx.item()}")
print(f"最高相似度 Patch 坐标: {coords[max_sim_idx].cpu().tolist()}\n") # Explicitly move to CPU for printing


# --- 4. 基于坐标的滑动窗口 ---
print("--- 4. 基于坐标的滑动窗口 ---")
WINDOW_SIZE = 3
STRIDE = 3

# 为了构建 r_map, c_map 和 patch_grid，坐标最好在CPU上处理
coords_cpu = coords.cpu()
unique_r_coords = torch.unique(coords_cpu[:, 0])
unique_c_coords = torch.unique(coords_cpu[:, 1])

r_map = {r.item(): i for i, r in enumerate(unique_r_coords)}
c_map = {c.item(): i for i, c in enumerate(unique_c_coords)}

grid_rows = len(unique_r_coords)
grid_cols = len(unique_c_coords)

# patch_grid 在 CPU 上创建和操作
patch_grid = torch.full((grid_rows, grid_cols), -1, dtype=torch.long, device='cpu')
for i in range(coords_cpu.shape[0]):
    r_idx = r_map[coords_cpu[i, 0].item()]
    c_idx = c_map[coords_cpu[i, 1].item()]
    if patch_grid[r_idx, c_idx] == -1 :
        patch_grid[r_idx, c_idx] = i

print(f"推断的网格大小: {grid_rows} 行 x {grid_cols} 列 (在CPU上构建)")

window_scores = []
for r_start in range(0, grid_rows - WINDOW_SIZE + 1, STRIDE):
    for c_start in range(0, grid_cols - WINDOW_SIZE + 1, STRIDE):
        window_patch_indices_on_grid = patch_grid[r_start:r_start+WINDOW_SIZE, c_start:c_start+WINDOW_SIZE] # CPU tensor
        valid_patch_original_indices = window_patch_indices_on_grid[window_patch_indices_on_grid != -1].tolist() # Python list of CPU ints
        
        if not valid_patch_original_indices:
            continue
        
        # cos_similarities 在 GPU 上，用 CPU 索引列表进行索引是允许的
        current_window_total_sim = cos_similarities[valid_patch_original_indices].sum().item() # .sum() on GPU, .item() moves to CPU
        window_scores.append({
            'score': current_window_total_sim,
            'patch_indices': valid_patch_original_indices, # list of CPU ints
            'grid_pos': (r_start, c_start)
        })

if not window_scores:
    print("错误：没有形成任何有效的窗口。请检查坐标分布、窗口大小和步长设置。")
else:
    print(f"总共生成 {len(window_scores)} 个窗口。\n")


# --- 5. 窗口筛选 ---
print("--- 5. 窗口筛选 ---")
if window_scores:
    window_scores.sort(key=lambda x: x['score'], reverse=True)
    num_windows_to_keep = max(1, int(len(window_scores) * 0.5)) if len(window_scores) > 0 else 0
    selected_windows = window_scores[:num_windows_to_keep]
    
    if not selected_windows:
        print("警告：筛选后没有剩余窗口。")
    else:
        print(f"筛选后保留 {len(selected_windows)} 个窗口。")
        print(f"得分最高的窗口得分: {selected_windows[0]['score'] if selected_windows else 'N/A'}\n")
else:
    selected_windows = []
    print("由于没有窗口生成，筛选步骤跳过。\n")


# --- 6. 窗口内自注意力 ---
print("--- 6. 窗口内自注意力 ---")
embed_dim = feature_dim
num_heads = 8
self_attention_module = SelfAttentionLayer(embed_dim, num_heads).to(device) # 模型移动到设备
all_attended_patch_features = []

if not selected_windows:
    print("没有选中的窗口，跳过自注意力步骤。\n")
else:
    for i, window_info in enumerate(selected_windows):
        patch_indices_in_window = window_info['patch_indices'] # list of CPU ints
        if not patch_indices_in_window:
            print(f"警告: 选中的窗口 {i} 不包含任何 patch 索引，跳过。")
            continue
        
        # bag_feats 在 GPU 上，用 CPU 索引列表索引，结果仍在 GPU 上
        current_window_feats = bag_feats[patch_indices_in_window] 
        current_window_feats_batched = current_window_feats.unsqueeze(0) # Still on GPU
        
        attended_feats_in_window = self_attention_module(current_window_feats_batched).squeeze(0) # Output on GPU
        all_attended_patch_features.append(attended_feats_in_window)
        if i < 2:
             print(f"窗口 {i}: 包含 {len(patch_indices_in_window)} 个 patch, 自注意力后特征形状: {attended_feats_in_window.shape}, device: {attended_feats_in_window.device}")

if not all_attended_patch_features:
    print("警告：经过自注意力处理后没有特征可用于MIL聚合。\n")
else:
    aggregated_attended_features = torch.cat(all_attended_patch_features, dim=0) # Concatenates GPU tensors, result on GPU
    print(f"\n所有窗口自注意力后的特征聚合形状: {aggregated_attended_features.shape}, device: {aggregated_attended_features.device}\n")


# --- 7. MIL 聚合 ---
print("--- 7. MIL 聚合 ---")
if not all_attended_patch_features:
    print("没有特征送入MIL聚合器，跳过此步骤。")
    new_image_feature = torch.zeros(1, feature_dim, device=device) # Default on specified device
    attention_scores_mil = None
else:
    mil_input_dim = feature_dim
    mil_hidden_dim = 256
    mil_output_dim = feature_dim
    mil_aggregator = AttentionMIL(input_dim=mil_input_dim, hidden_dim=mil_hidden_dim, output_dim=mil_output_dim).to(device) # 模型移动到设备
    
    # aggregated_attended_features 已经在 device 上
    new_image_feature, attention_scores_mil = mil_aggregator(aggregated_attended_features) # Output on GPU

# ... (您之前的代码) ...

print(f"通过 MIL 聚合器得到的新图片特征形状: {new_image_feature.shape if new_image_feature is not None else 'N/A'}, device: {new_image_feature.device if new_image_feature is not None else 'N/A'}")
if new_image_feature is not None:
    print(f"新图片特征 (前10个值): {new_image_feature.squeeze().cpu()[:10].tolist()}") # Move to CPU for printing
    
    # --- 保存最终的图片特征 ---
    output_feature_path = '/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/MUSK-MIL-feature/final_image_feature.pt'
    # 建议在保存前将其移至CPU，以提高可移植性
    torch.save(new_image_feature.cpu(), output_feature_path)
    print(f"\n最终的图片特征已保存到: {output_feature_path}")

if attention_scores_mil is not None:
    print(f"MIL 注意力权重形状: {attention_scores_mil.shape}, device: {attention_scores_mil.device}")

print("\n处理流程完成！")