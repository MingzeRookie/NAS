#!/bin/bash

# 设置环境变量
# 设置CUDA内存分配器，帮助处理内存碎片问题
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export CUDA_VISIBLE_DEVICES=2,3,4,5  # 使用CUDA 2,3,4,5设备
export PYTHONPATH=$PYTHONPATH:$(pwd)  # 添加当前目录到Python路径

# 设置路径
OUTPUT_DIR="./output/univ2_qlora_multi_gpu"
CONFIG_PATH="/remote-home/share/lisj/Workspace/SOTA_NAS/preprocess/univ2_qlora_ddp/configs/qlora_config_ultrastable_multi.yaml"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p ./configs

# 如果配置文件不存在，则复制一份
if [ ! -f $CONFIG_PATH ]; then
    cp ./configs/qlora_config_ultrastable_multi.yaml $CONFIG_PATH
    # 更新配置文件中的batch_size
    sed -i 's/batch_size_per_gpu: 4/batch_size_per_gpu: 4/g' $CONFIG_PATH
    sed -i 's/output_dir: \.\/output\/univ2_qlora_ultrastable/output_dir: \.\/output\/univ2_qlora_multi_gpu/g' $CONFIG_PATH
fi
# 使用torchrun替代torch.distributed.launch
torchrun \
  --nproc_per_node=4 \
  --master_port=29501 \
  train.py \
  --config $CONFIG_PATH \
  --output_dir $OUTPUT_DIR \
  --batch_size 4 \
  --epochs 25 \
  --seed 42 \
  --num_workers 4 \
  --save_freq 2 \
  --eval_freq 1 \
  --distributed \
  2>&1 | tee $OUTPUT_DIR/training.log