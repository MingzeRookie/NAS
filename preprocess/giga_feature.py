import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import glob
import sys
import timm
import re

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 添加Lambda转换来确保图像为 RGB (3 通道)
transform = transforms.Compose([
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),  # 保证是RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

local_dir = "/remote-home/share/zhangyuanyuan/prov-gigapath/ckpt/"
model = timm.create_model("vit_giant_patch14_dinov2", patch_size=16, num_classes=0, pretrained=False)
state_dict = torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu")
pos_embed = state_dict["pos_embed"]
new_pos_embed = torch.zeros(1, 1025, 1536)
new_pos_embed[:, :pos_embed.shape[1], :] = pos_embed
state_dict["pos_embed"] = new_pos_embed
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()

root_dir = '../datasets/patches'
save_dir = '../datasets/GIGA-feature'
os.makedirs(save_dir, exist_ok=True)

coord_pattern = re.compile(r"(\d+)_(\d+)$")

for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    csv_file_path = os.path.join(save_dir, f"{subfolder}.csv")
    processed_coords = set()

    if os.path.exists(csv_file_path):
        df_existing = pd.read_csv(csv_file_path)
        processed_coords = set(df_existing["coord"].astype(str))

    feature_list = []
    png_files = glob.glob(os.path.join(subfolder_path, "*.png"))

    for png_file in tqdm(png_files, desc=f"Processing {subfolder}"):
        img_name = os.path.basename(png_file).split('.')[0]
        match = coord_pattern.match(img_name)
        coord = f"({match.group(1)},{match.group(2)})" if match else img_name

        if coord in processed_coords:
            continue
        
        try:
            img = Image.open(png_file).convert('RGB')
            img = transform(img)
        except Exception as e:
            print(f"Error processing {png_file}: {e}")
            continue
    
        img = img.unsqueeze(0).to(device)  # Move to GPU

        with torch.no_grad():
            feature = model(img).cpu().squeeze().numpy()  # Directly call the model to get features

        feature_list.append([coord] + feature.tolist())

    if feature_list:
        num_features = len(feature_list[0]) - 1
        column_names = ["coord"] + [str(i) for i in range(num_features)]
        df_new = pd.DataFrame(feature_list, columns=column_names)

        if os.path.exists(csv_file_path):
            df_new.to_csv(csv_file_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(csv_file_path, index=False)

print("特征提取和保存完成。")
