#!/bin/bash

# 设置环境变量
export PYTHONPATH="$PYTHONPATH:$(pwd)"  # 确保dinov2包在Python路径中
export CUDA_VISIBLE_DEVICES=2,3,4,5  # 使用的GPU设备
export MASTER_PORT=29500  # 主端口
export HF_TOKEN="YOUR_HF_TOKEN"  # 替换为您的HuggingFace token

# 配置参数
CONFIG_FILE="configs/uni_qlora.yaml"
OUTPUT_DIR="output/uni_qlora"
DATASET_PATH="/path/to/dataset"  # 替换为您的数据集路径
NUM_GPUS=4  # 使用的GPU数量

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 定义训练函数
train() {
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        dinov2/run/train.py \
        --config-file $CONFIG_FILE \
        --output-dir $OUTPUT_DIR \
        student.hf_token=$HF_TOKEN \
        train.dataset_path=$DATASET_PATH
}

# 定义帮助函数
print_help() {
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  train             运行训练"
    echo "  eval              运行评估"
    echo "  check             检查环境和配置"
    echo "  help              显示此帮助信息"
    echo ""
}

# 定义环境检查函数
check_env() {
    echo "检查环境和配置..."
    
    # 检查Python版本
    python --version
    
    # 检查PyTorch版本
    python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
    
    # 检查CUDA可用性
    python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
    python -c "import torch; print(f'CUDA设备数: {torch.cuda.device_count()}')"
    
    # 检查必要的包
    python -c "import transformers, peft, bitsandbytes, timm; print('所有必要的包都已安装')"
    
    # 检查huggingface token
    if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" == "YOUR_HF_TOKEN" ]; then
        echo "警告: HF_TOKEN未设置或使用了默认值"
    else
        echo "HF_TOKEN已设置"
    fi
    
    # 检查数据集路径
    if [ ! -d "$DATASET_PATH" ]; then
        echo "警告: 数据集路径 $DATASET_PATH 不存在"
    else
        echo "数据集路径: $DATASET_PATH"
    fi
    
    # 检查配置文件
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "错误: 配置文件 $CONFIG_FILE 不存在"
    else
        echo "配置文件: $CONFIG_FILE"
    fi
    
    # 检查输出目录
    echo "输出目录: $OUTPUT_DIR"
}

# 定义评估函数
eval() {
    echo "运行评估..."
    python dinov2/run/eval/linear.py \
        --config-file $CONFIG_FILE \
        --pretrained-weights $OUTPUT_DIR/checkpoint.pth \
        --train-dataset ImageNet:split=TRAIN:root=$DATASET_PATH \
        --val-dataset ImageNet:split=VAL:root=$DATASET_PATH
}

# 根据参数执行相应操作
case "$1" in
    train)
        train
        ;;
    eval)
        eval
        ;;
    check)
        check_env
        ;;
    help)
        print_help
        ;;
    *)
        if [ -z "$1" ]; then
            echo "未提供参数，默认执行训练"
            train
        else
            echo "未知参数: $1"
            print_help
            exit 1
        fi
        ;;
esac