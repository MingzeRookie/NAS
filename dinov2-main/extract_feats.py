import os
import argparse
from typing import Any, List, Optional, Tuple
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import torch
from dinov2.eval.setup import build_model_for_eval
import dinov2.utils.utils as dinov2_utils
from dinov2.utils.config import setup, get_cfg_from_args

def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    return parser

# def load_dino_model(config,pretrained_weights):
#     """

#     """
#     model = build_model_for_eval(config, only_teacher=True)
#     dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
#     model.eval()
#     model.cuda()
#     return model
    

def main(model):
    """
    Extract features from images using a DINOv2 model.
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
    # 加载模型
    # model = load_dino_model(args.config_file, args.pretrained_weights)
    # model.eval()
    # device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    
    # local_dir = "/remote-home/share/lisj/Workspace/SOTA_NAS/preprocess/dini/output/2025-04-14"
    # ckpt_path = os.path.join(local_dir, "checkpoint_step_3100.pth")
    root_dir = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/patches/train"
    out_dir = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/NASE-finetuned"
    os.makedirs(out_dir, exist_ok=True)
    
    for slide_id in tqdm(os.listdir(root_dir), desc="Slides", unit="slide"):
        slide_dir = os.path.join(root_dir, slide_id)
        if not os.path.isdir(slide_dir):
            continue

        out_path = os.path.join(out_dir, f"{slide_id}.pt")
        if os.path.exists(out_path):
            tqdm.write(f"[Skipped] {slide_id} 已处理")
            continue

        # features = {}
        pngs = [fn for fn in os.listdir(slide_dir) if fn.lower().endswith(".png")]
        features, coords = [], []
        
        for fn in tqdm(pngs, desc=slide_id, leave=False, unit="patch"):
            img = Image.open(os.path.join(slide_dir, fn)).convert("RGB")
            x = transform(img).unsqueeze(0).cuda()
            with torch.inference_mode():
                feat = model(x)
            feat = feat.squeeze(0).cpu()
            coords.append(os.path.splitext(fn)[0])
            features.append(feat)
        file = {
            "bag_feats": torch.stack(features,dim=0),
            "coords": coords
        }
        torch.save(file, out_path)
        tqdm.write(f"[Saved]   {slide_id}.pt ({len(features)} patches)")
        
        
if __name__ == "__main__":     
    # 使用示例
    description = "DINOv2 extract features"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    config = get_cfg_from_args(args)
    model = build_model_for_eval(config, args.pretrained_weights)
    main(model)
