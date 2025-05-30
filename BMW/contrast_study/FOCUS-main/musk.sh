#!/bin/bash

# 日志目录，假设脚本在 FOCUS-main 目录下运行
log_dir='logs/' 
# 确保日志目录存在 (如果脚本是从 FOCUS-main 目录外运行，请使用绝对路径或确保 FOCUS-main/logs 存在)
mkdir -p $log_dir # 如果 musk.sh 和 logs/ 都在 FOCUS-main/ 下，这个相对路径是OK的

task='task_musk' 
model='FOCUS'
feature='musk' 
device=4 # 您的GPU ID

export CUDA_VISIBLE_DEVICES=$device
# exp 变量主要用于结果目录的组织
exp="${model}/${feature}" # 使用引号以防变量中包含特殊字符
echo "Task: ${task}, Model: ${exp}, GPU No.: ${device}"

# 确保 main.py 是您的主训练脚本
# 并确保所有路径都是正确的

nohup python main.py \
    --seed 1 \
    --drop_out \
    --early_stopping \
    --lr 1e-4 \
    --bag_loss ce \
    --task "${task}" \
    --results_dir "results/${exp}/" \
    --exp_code "${task}_single_run" \
    --model_type "${model}" \
    --mode transformer \
    --log_data \
    --train_csv "/remote-home/share/lisj/Workspace/SOTA_NAS/BMW/datasets/train_inflammation_labels.csv" \
    --val_csv "/remote-home/share/lisj/Workspace/SOTA_NAS/BMW/datasets/val_inflammation_labels.csv" \
    --data_folder_l "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/MUSK-feature" \
    --max_context_length 128 \
    --text_prompt_path "/remote-home/share/lisj/Workspace/SOTA_NAS/BMW/contrast_study/FOCUS-main/text_prompt/MUSK_text_prompts.csv" \
    --window_size 16 \
    --sim_threshold 0.85 \
    --feature_dim 1024 \
    --prototype_number 16
    > "${log_dir}${task}_${model}_${feature}_single_run.log" 2>&1 &

echo "FOCUS training started with task: ${task}. Log: ${log_dir}${task}_${model}_${feature}_single_run.log"