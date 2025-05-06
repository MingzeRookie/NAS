# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import timm
import torch 
from . import vision_transformer as vits


logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim

# def build_model(args, only_teacher=False, img_size=224):
#     timm_kwargs = {
#             'model_name': 'vit_giant_patch14_224',
#             'img_size': 224, 
#             'patch_size': 14, 
#             'depth': 24,
#             'num_heads': 24,
#             'init_values': 1e-5, 
#             'embed_dim': 1536,
#             'mlp_ratio': 2.66667*2,
#             'num_classes': 0, 
#             'no_embed_class': True,
#             'mlp_layer': timm.layers.SwiGLUPacked, 
#             'act_layer': torch.nn.SiLU, 
#             'reg_tokens': 8, 
#             'dynamic_img_size': True
#     }
#     student = timm.create_model(**timm_kwargs)
#     teacher = timm.create_model(**timm_kwargs)
#     embed_dim = student.embed_dim
#     return student, teacher, embed_dim
    

def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)
