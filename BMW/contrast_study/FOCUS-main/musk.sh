log_dir='FOCUS-main/logs/' # 例如 FOCUS-main/logs/
task='task_musk' # 与您在 main.py 中定义的 task 名称一致
model='FOCUS'
feature='musk' # 反映您使用的是musk特征
device=4 # 您的GPU ID

export CUDA_VISIBLE_DEVICES=$device
exp=$model"/"$feature
echo "Task: "$task", Model: "$exp", GPU No.:"$device

# 假设您的主训练脚本在 FOCUS-main 目录下名为 main.py
nohup python main.py \
    --seed 1 \
    --drop_out \
    --early_stopping \
    --lr 1e-4 \
    --bag_loss ce \
    --task $task \
    --results_dir 'results/'$model'/'$feature'/' \
    --exp_code $task"_single_run" \
    --model_type $model \
    --mode transformer \
    --log_data \
    # 数据集路径
    --train_csv "/remote-home/share/lisj/Workspace/SOTA_NAS/BMW/datasets/train_inflammation_labels.csv" \ # 替换为实际路径
    --val_csv "/remote-home/share/lisj/Workspace/SOTA_NAS/BMW/datasets/val_inflammation_labels.csv" \   # 替换为实际路径
    # --test_csv "/path/to/your/test_labels.csv" # 如果有单独测试集，取消注释并设置路径

    # Musk 特征路径 (用于 features_l)
    --data_folder_l "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/MUSK-feature" \ # 替换为实际路径
    # --data_folder_s "" # 保持为空或不设置此参数，让 features_s 成为占位符

    # 文本提示
    --text_prompt_path "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/MUSK-feature/MUSK-text-feature/text_feature.pt" \ # 替换为您创建的文本提示文件路径 (在FOCUS-main下)

    # FOCUS 模型特定参数
    --window_size 16 \
    --sim_threshold 0.85 \
    --feature_dim 1024 \
    --max_context_length 128 \
    # --prototype_number 16 # 如果 args.prototype_number 被 model_FOCUS 或其依赖使用，则保留

    > $log_dir$task"_"$model"_"$feature"_single_run.log" 2>&1 &