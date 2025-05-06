import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import glob
import sys
import re  # 用于解析坐标

# 添加 encoder 目录到 sys.path
sys.path.append("/remote-home/share/lisj/Workspace/SOTA_NAS/encoder")

from conch.open_clip_custom.factory import create_model_from_pretrained

# ####  CONCH  #####
model, transform = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="/remote-home/share/zhangyuanyuan/CONCH/ckpt/pytorch_model.bin")
model.cuda()
model.eval()

root_dir = '../datasets/core/patches'
save_dir = '../datasets/core/CONCH-feature'
os.makedirs(save_dir, exist_ok=True)

# 用正则表达式匹配 `x_y.png` 形式的文件名
coord_pattern = re.compile(r"(\d+)_(\d+)$")

for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    csv_file_path = os.path.join(save_dir, f"{subfolder}.csv")
    processed_coords = set()

    # 如果 CSV 已存在，读取已处理的坐标
    if os.path.exists(csv_file_path):
        df_existing = pd.read_csv(csv_file_path)
        processed_coords = set(df_existing["coord"].astype(str))  # 存储已处理的坐标

    feature_list = []
    png_files = glob.glob(os.path.join(subfolder_path, "*.png"))

    for png_file in tqdm(png_files, desc=f"Processing {subfolder}"):
        img_name = os.path.basename(png_file).split('.')[0]  # 获取不带扩展名的文件名
        match = coord_pattern.match(img_name)
        coord = f"({match.group(1)},{match.group(2)})" if match else img_name  # 解析坐标

        if coord in processed_coords:
            continue  # 跳过已处理的图片

        img = Image.open(png_file)
        img = transform(img)  # 预处理
        img = img.unsqueeze(0).cuda()  # 添加批次维度并移动到 GPU

        with torch.no_grad():
            feature = model.encode_image(img, proj_contrast=False, normalize=False).cpu().squeeze().numpy()

        feature_list.append([coord] + feature.tolist())  # 第一列是坐标，后续列是特征

    if feature_list:
        num_features = len(feature_list[0]) - 1
        column_names = ["coord"] + [str(i) for i in range(num_features)]
        df_new = pd.DataFrame(feature_list, columns=column_names)

        if os.path.exists(csv_file_path):
            df_new.to_csv(csv_file_path, mode='a', header=False, index=False)  # 追加模式
        else:
            df_new.to_csv(csv_file_path, index=False)

print("特征提取和保存完成。")
