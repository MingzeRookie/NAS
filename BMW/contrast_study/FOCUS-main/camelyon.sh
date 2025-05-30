log_dir='logs/'
task='camelyon'
shots=16
folds=10
model='FOCUS'
feature='conch'
device=0

export CUDA_VISIBLE_DEVICES=$device
exp=$model"/"$feature
echo "Task: "$task", Shots: "$shots", "$exp", GPU No.:"$device
nohup python main.py \
    --seed 1 \
    --drop_out \
    --early_stopping \
    --lr 1e-4 \
    --k $folds \
    --label_frac 1 \
    --bag_loss ce \
    --task "task_camelyon_subtyping" \
    --results_dir 'results/'$model'/'$feature'/' \
    --exp_code $task"_"$shots"shots_"$folds"folds" \
    --model_type $model \
    --mode transformer \
    --log_data \
    --data_root_dir 'path/to/your/data' \
    --data_folder_s 'path/to/your/low-resolution/feature' \
    --data_folder_l 'path/to/your/high-resolution/feature' \
    --split_dir $task"_"$shots"shots_"$folds"folds" \
    --text_prompt_path 'text_prompt/CAMELYON_two_scale_text_prompt.csv' \
    --prototype_number 16 > $log_dir$task"_"$model"_"$shots"shots_"$folds"folds_"$feature".log" 2>&1 &