import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import math

def calculate_similarities(
    text_feature_path: str,
    combined_feature_path: str,
):
    """计算并返回所有 patch 的坐标和对应的余弦相似度分数。"""
    try:
        text_feature = torch.load(text_feature_path)
        data_dict = torch.load(combined_feature_path)
        patch_features = data_dict['bag_feats']
        patch_coords = data_dict['coords']
    except Exception as e:
        print(f"❌ 加载文件 {combined_feature_path} 时出错: {e}")
        return None, None

    device = torch.device("cpu")
    text_feature = text_feature.to(device).to(torch.float32)
    patch_features = patch_features.to(device).to(torch.float32)
    patch_coords = patch_coords.to(device)

    if text_feature.dim() == 1:
        text_feature = text_feature.unsqueeze(0)

    text_feature_norm = F.normalize(text_feature, p=2, dim=1)
    patch_features_norm = F.normalize(patch_features, p=2, dim=1)
    
    cosine_similarities = torch.mm(text_feature_norm, patch_features_norm.t()).squeeze()
    
    return patch_coords, cosine_similarities


def create_attention_grid(
    patch_dir: str,
    coords: torch.Tensor,
    scores: torch.Tensor,
    patch_size: int,
    output_path: str,
    top_k: int = 100,
    padding: int = 10
):
    """
    【修改版】将分数最高的K个patch以网格形式平铺，显示原色图像。
    """
    coords_np = coords.numpy()
    scores_np = scores.numpy()

    # 1. 找出分数最高的 Top K 个 patch 的索引
    if len(scores_np) < top_k:
        print(f"⚠️ patch数量 ({len(scores_np)}) 小于 top_k ({top_k})，将显示所有 patch。")
        top_k = len(scores_np)

    top_scores, top_indices = torch.topk(torch.from_numpy(scores_np), k=top_k)
    top_scores_np = top_scores.numpy()

    # 2. 计算网格布局
    grid_cols = math.ceil(math.sqrt(top_k))
    grid_rows = math.ceil(top_k / grid_cols)
    
    cell_size = patch_size + padding
    canvas_w = grid_cols * cell_size + padding
    canvas_h = grid_rows * cell_size + padding
    
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240 # 浅灰色背景

    # 3. 遍历 Top K patch 并将它们放入网格
    for i in range(top_k):
        idx = top_indices[i]
        x, y = coords_np[idx]
        
        patch_filename = f"{x}_{y}.png"
        patch_path = os.path.join(patch_dir, patch_filename)
        
        if os.path.exists(patch_path):
            patch_img = cv2.imread(patch_path)
            if patch_img is None or patch_img.shape[0] != patch_size or patch_img.shape[1] != patch_size:
                patch_img = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * 128
                cv2.putText(patch_img, "N/A", (50, 112), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        else:
            patch_img = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * 128
            cv2.putText(patch_img, "NoFile", (30, 112), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)


        # 【核心修改】不再进行颜色叠加，直接使用原图
        # 在 patch 左上角写上排名和分数
        rank_text = f"#{i+1}"
        score_text = f"S:{top_scores_np[i]:.3f}" # 显示原始分数，更直观
        # 添加一个黑色背景以便于看清白色文字
        cv2.rectangle(patch_img, (0, 0), (90, 45), (0,0,0), -1)
        cv2.putText(patch_img, rank_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(patch_img, score_text, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # 计算在网格中的位置
        row = i // grid_cols
        col = i % grid_cols
        paste_x = col * cell_size + padding
        paste_y = row * cell_size + padding
        
        canvas[paste_y:paste_y+patch_size, paste_x:paste_x+patch_size] = patch_img

    # 4. 保存结果
    cv2.imwrite(output_path, canvas)
    

if __name__ == '__main__':
    # --- ❗请配置以下4个路径 ---

    # 1. 固定的文本特征文件路径
    TEXT_FEATURE_PATH = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/MUSK-feature/MUSK-text-feature/text_feature.pt"
    
    # 2. 存放所有切片特征 .pt 文件的文件夹
    FEATURES_DIR = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/MUSK-feature/"
    
    # 3. 存放所有 pngs_{slide_id} 文件夹的根目录
    PNG_ROOT_DIR = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/patches/train"

    # 4. 用于保存所有生成的注意力图的输出文件夹
    OUTPUT_DIR = "/remote-home/share/lisj/Workspace/SOTA_NAS/BMW/study_results/attention_map"

    # --- 可选参数 ---
    PATCH_SIZE = 224
    TOP_K_PATCHES = 100 # 每张图显示 top 100 个 patch

    # --- 批量处理主逻辑 ---
    
    # 确保输出文件夹存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有待处理的 .pt 文件
    feature_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith('.pt') and f != 'text_feature.pt']
    
    print(f"扫描到 {len(feature_files)} 个切片特征文件，将开始批量处理...")

    for filename in feature_files:
        slide_id = os.path.splitext(filename)[0]
        print(f"\n--- 正在处理切片: {slide_id} ---")

        # 构建当前切片所需的所有路径
        combined_feature_path = os.path.join(FEATURES_DIR, filename)
        # 假设png文件夹名为 pngs_{slide_id}
        patch_dir = os.path.join(PNG_ROOT_DIR, f"{slide_id}")
        output_path = os.path.join(OUTPUT_DIR, f"{slide_id}_attention_map.png")

        # 检查对应的png文件夹是否存在
        if not os.path.isdir(patch_dir):
            print(f"❌ 警告: 未找到对应的PNG文件夹，跳过此切片: {patch_dir}")
            continue

        # 1. 计算相似度
        all_coords, all_scores = calculate_similarities(
            text_feature_path=TEXT_FEATURE_PATH,
            combined_feature_path=combined_feature_path,
        )
        
        # 2. 如果成功，则生成注意力图
        if all_coords is not None and all_scores is not None:
            create_attention_grid(
                patch_dir=patch_dir,
                coords=all_coords,
                scores=all_scores,
                patch_size=PATCH_SIZE,
                top_k=TOP_K_PATCHES,
                output_path=output_path
            )
            print(f"✅ 注意力图已保存到: {output_path}")

    print("\n--- 所有任务处理完毕！ ---")