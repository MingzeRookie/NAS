import torch
from timm.models import create_model
import sys, os, glob, re
sys.path.insert(0, "/remote-home/share/lisj/Workspace/SOTA_NAS/encoder/musk/MUSK-main")
from musk import utils, modeling
from PIL import Image
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torchvision
from tqdm import tqdm
import torch.multiprocessing as mp

# —— 配置区 —— 
GPU_LIST   = [1, 2, 3, 5]  # 要使用的 4 张卡
TRAIN_ROOT = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/patches/train"
OUT_ROOT   = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/MUSK-feature"
CKPT_PATH  = "/remote-home/share/lisj/Workspace/SOTA_NAS/encoder/musk/checkpoint/model.safetensors"
MODEL_NAME = "musk_large_patch16_384"
IMG_SIZE   = 384

os.makedirs(OUT_ROOT, exist_ok=True)

# 预编译正则，仅匹配 "数字_数字.png"
coord_re = re.compile(r'^(\d+)_(\d+)\.png$')

# 图像预处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE, interpolation=3, antialias=True),
    torchvision.transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD
    )
])

def worker(local_rank, case_lists, gpu_list):
    gpu_id = gpu_list[local_rank]
    device = torch.device(f"cuda:{gpu_id}")

    # 加载模型到对应 GPU
    model = create_model(MODEL_NAME).to(device, torch.float16).eval()
    utils.load_model_and_may_interpolate(CKPT_PATH, model, 'model|module', '')
    model.eval()

    my_cases = case_lists[local_rank]
    for case in my_cases:
        src_dir = os.path.join(TRAIN_ROOT, case)
        out_path = os.path.join(OUT_ROOT, f"{case}.pt")
        if os.path.exists(out_path):
            print(f"[GPU{gpu_id}] skip {case} (exists)")
            continue

        feats_list = []
        coords_list = []

        # 只匹配 *_*.png
        img_paths = sorted(
            glob.glob(os.path.join(src_dir, "*_*.png")),
            key=lambda p: tuple(
                map(int, os.path.basename(p).replace('.png','').split('_'))
            )
        )
        pbar = tqdm(img_paths, desc=f"[GPU{gpu_id}] {case}", ncols=80, leave=False)
        for img_path in pbar:
            fname = os.path.basename(img_path)
            m = coord_re.match(fname)
            if not m:
                # 跳过不符合格式的文件
                continue
            x, y = map(int, m.groups())

            # 读取 & 预处理
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device, torch.float16)

            # 前向提取
            with torch.inference_mode():
                emb = model(
                    image=img_tensor,
                    with_head=False,
                    out_norm=True
                )[0].squeeze(0).cpu()  # (D,)

            feats_list.append(emb)
            coords_list.append([x, y])

        if not feats_list:
            print(f"[GPU{gpu_id}] no valid patches in {case}, skipped")
            continue

        # 堆叠并保存
        bag_feats = torch.stack(feats_list, dim=0)             # (N_patches, D)
        coords    = torch.tensor(coords_list, dtype=torch.int) # (N_patches, 2)

        torch.save({'bag_feats': bag_feats, 'coords': coords}, out_path)
        print(f"[GPU{gpu_id}] saved {bag_feats.shape[0]} patches → {out_path}")

if __name__ == "__main__":
    # 收集所有 case 并轮询分配给每个进程
    all_cases = [
        d for d in os.listdir(TRAIN_ROOT)
        if os.path.isdir(os.path.join(TRAIN_ROOT, d))
    ]
    num_procs = len(GPU_LIST)
    per_proc = [[] for _ in range(num_procs)]
    for idx, case in enumerate(all_cases):
        per_proc[idx % num_procs].append(case)

    mp.spawn(
        worker,
        args=(per_proc, GPU_LIST),
        nprocs=num_procs,
        join=True
    )
