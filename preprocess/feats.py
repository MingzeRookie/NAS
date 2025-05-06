import glob
import os

import pandas as pd
import timm
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from feat.conch.open_clip_custom import create_model_from_pretrained

####  CONCH  #######
model, transform = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="../CONCH/ckpt/pytorch_model.bin")
model.cuda()
model.eval()  # 设置为评估模式


####  Prov-gigapath  ########
# 定义图像变换
# transform = transforms.Compose(
#     [
#         transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
#         transforms.CenterCrop(518),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ]
# )
# local_dir = "/remote-home/share/zhangyuanyuan/prov-gigapath/ckpt/"
# model = timm.create_model(" vit_giant_patch14_dinov2", patch_size=16, num_classes=0, pretrained=False)
# # 加载权重
# state_dict = torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu")
# # 手动调整 pos_embed 的形状
# pos_embed = state_dict["pos_embed"]
# # pos_embed.shape = [1, 197, 1536] (在checkpoint中)
# # 需要将其调整为 [1, 1025, 1536] (在当前模型中)
# # 这里假设你只需要取前1025个位置，您可能需要根据具体情况进行调整
# new_pos_embed = torch.zeros(1, 1025, 1536)  # 初始化新的位置嵌入
# new_pos_embed[:, :pos_embed.shape[1], :] = pos_embed  # 复制旧的嵌入
# # 替换旧的 pos_embed
# state_dict["pos_embed"] = new_pos_embed
# # 加载调整后的 state_dict
# model.load_state_dict(state_dict, strict=True)
#-----------------------------------------------------------------------------------------------------


# 根目录
root_dir = '/remote-home/share/lisj/Unitopatho/unitopath_patches/test'
# 保存的目标目录
save_dir = '/remote-home/share/zhangyuanyuan/patch_feats/Test/Conch'
# 子文件夹名称
subfolder_names = ['HP', 'NORM', 'TA.HG', 'TA.LG', 'TVA.HG', 'TVA.LG']

# 遍历每个子文件夹
for subfolder in subfolder_names:
    subfolder_path = os.path.join(root_dir, subfolder)

    # 为每个subfolder_path创建一个对应的文件夹来存储CSV文件
    subfolder_save_dir = os.path.join(save_dir, subfolder)
    os.makedirs(subfolder_save_dir, exist_ok=True)  # 创建子文件夹保存CSV文件

    # 遍历该子文件夹下的第一级子文件夹
    for class_dir in glob.glob(os.path.join(subfolder_path, "*")):
        if os.path.isdir(class_dir):
            # 存储特征的列表
            feature_list = []

            # 获取该第一级子文件夹中所有符合条件的jpg文件，筛选包含"20x"的文件名
            jpg_files = glob.glob(os.path.join(class_dir, "*20x*.jpg"))

            # 打印每个class_dir中的jpg文件
            print(f"Checking files in: {class_dir}")
            print("Found JPG files:")

            # 遍历每个符合条件的jpg文件
            for jpg_file in tqdm(jpg_files):
                # 读取图像并应用转换
                # print(jpg_file)  # 打印符合条件的文件路径
                img = Image.open(jpg_file)
                img = preprocess(img).unsqueeze(0)
                img = img.cuda()  # 添加批次维度

                with torch.no_grad():
                    features  =  model.encode_image(img, proj_contrast=False, normalize=False).cpu().squeeze().numpy()  # 添加批次维度

                # 获取文件名的前缀
                img_name = os.path.basename(jpg_file).split('.')[0]  # 获取不带后缀的文件名
                feature_list.append([img_name] + features.tolist())  # 将文件名和特征合并为一行

            # 将特征保存为CSV文件，文件名为第一级子文件夹的名字
            if feature_list:  # 只在有特征时创建文件
                df = pd.DataFrame(feature_list)
                csv_file_path = os.path.join(subfolder_save_dir, f"{os.path.basename(class_dir)}.csv")  # 使用第一级子文件夹名作为CSV文件名
                df.to_csv(csv_file_path, index=False, header=False)  # 不写入行索引和列名

            # 打印该第一级子文件夹中提取的数据条数
            print(f"从 {class_dir} 提取的图片数量: {len(feature_list)}")  # 显示提取的数据条数

print("特征提取和保存完成。")