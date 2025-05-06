import os
import torch
from torchvision import transforms
import timm
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

local_dir = "/remote-home/share/lisj/Workspace/SOTA_NAS/preprocess/dini/output/2025-04-14"
ckpt_path = os.path.join(local_dir, "checkpoint_step_3100.pth")
root_dir = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/patches/train"
out_dir = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/UNI-featrue-finetuned"
os.makedirs(out_dir, exist_ok=True)

timm_kwargs = {
    'model_name': 'vit_giant_patch14_224',
    'img_size': 224,
    'patch_size': 14,
    'depth': 24,
    'num_heads': 24,
    'init_values': 1e-5,
    'embed_dim': 1536,
    'mlp_ratio': 2.66667 * 2,
    'num_classes': 0,
    'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked,
    'act_layer': torch.nn.SiLU,
    'reg_tokens': 8,
    'dynamic_img_size': True,
    'pretrained': False,
}
model = timm.create_model(**timm_kwargs)
# state_dict = torch.load(ckpt_path, map_location='cpu')
# model.load_state_dict(state_dict, strict=True)
# model = model.to(device)

ckpt = torch.load(ckpt_path, map_location='cpu')

if 'state_dict' in ckpt:
    raw_sd = ckpt['state_dict']
elif 'model_state_dict' in ckpt:
    raw_sd = ckpt['model_state_dict']
elif 'model' in ckpt:
    raw_sd = ckpt['model']
else:
    raw_sd = ckpt  

from collections import OrderedDict
clean_sd = OrderedDict()
for k, v in raw_sd.items():
    name = k.replace('module.', '')
    clean_sd[name] = v

model.load_state_dict(clean_sd, strict=False)
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
model.eval()

for slide_id in tqdm(os.listdir(root_dir), desc="Slides", unit="slide"):
    slide_dir = os.path.join(root_dir, slide_id)
    if not os.path.isdir(slide_dir):
        continue

    out_path = os.path.join(out_dir, f"{slide_id}.pt")
    if os.path.exists(out_path):
        tqdm.write(f"[Skipped] {slide_id} 已处理")
        continue

    features = {}
    pngs = [fn for fn in os.listdir(slide_dir) if fn.lower().endswith(".png")]
    for fn in tqdm(pngs, desc=slide_id, leave=False, unit="patch"):
        img = Image.open(os.path.join(slide_dir, fn)).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            feat = model(x)
        feat = feat.squeeze(0).cpu()
        key = os.path.splitext(fn)[0]
        features[key] = feat

    torch.save(features, out_path)
    tqdm.write(f"[Saved]   {slide_id}.pt ({len(features)} patches)")
