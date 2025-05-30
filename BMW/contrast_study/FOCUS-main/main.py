from __future__ import print_function
import argparse
import os
from utils.file_utils import save_pkl # 确保 utils 文件夹在 FOCUS-main 目录下
from utils.utils import * # 确保 utils 文件夹在 FOCUS-main 目录下
from utils.core_utils import train  # 确保 utils 文件夹在 FOCUS-main 目录下
from datasets.dataset_generic import Generic_MIL_Dataset # 确保 datasets 文件夹在 FOCUS-main 目录下
import torch
import pandas as pd
import numpy as np
import sys # <--- 添加这一行
print("Raw command-line arguments:", sys.argv) # <--- 添加这一行
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory (often not needed if data_folder_l provides full path)')
parser.add_argument('--data_folder_s', type=str, default=None, help='directory for shallow features (x_s), can be None or empty if x_s is unused/dummy' )
parser.add_argument('--data_folder_l', type=str, default=None, help='directory for deep features (x_l, your musk features)')
parser.add_argument('--max_epochs', type=int, default=200, help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001, typical for FOCUS might be 1e-4 or 2e-4)') # 调整了默认值以匹配常见实践
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0, for direct train/val, this is likely always 1.0)')
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
# K-fold arguments removed as per your request
# parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
# parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
# parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
# parser.add_argument('--split_dir', type=str, default=None) # Removed, not using external split files

parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard') # Keep if tensorboard is used in core_utils.train
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=True, help='enable dropout (p=0.25)') # Defaulting to True as per camelyon.sh
parser.add_argument('--model_type', type=str, choices=['ViLa_MIL', 'FOCUS', 'mil'], default='FOCUS', help='model type') # Added 'mil' if it's an option in core_utils
parser.add_argument('--mode', type=str, choices=['transformer'], default='transformer', help='Attention mode for FOCUS or other models')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'focal'], default='ce') # 'focal' might need FocalLoss from loss_utils.py
parser.add_argument('--task', type=str, help='task identifier, e.g., your_musk_task_name') # This will be used to set n_classes and label_dict

# Text prompt related (for FOCUS)
parser.add_argument("--text_prompt_path", type=str, default=None, help='Path to the CSV file containing text prompts')

# FOCUS model specific parameters (consolidated and from your additions)
parser.add_argument('--window_size', type=int, default=16, help='Window size for FOCUS model')
parser.add_argument('--sim_threshold', type=float, default=0.85, help='Similarity threshold for FOCUS model')
parser.add_argument('--feature_dim', type=int, default=1024, help='Dimension of input image features (x_l for FOCUS, your musk features)')
parser.add_argument('--max_context_length', type=int, default=128, help='Max context length for FOCUS model token selection')
parser.add_argument("--prototype_number", type=int, default=16, help='Prototype number (used by some models like ViLa_MIL, check if FOCUS uses it via args in core_utils.train)')


# Paths for direct train/val CSVs
parser.add_argument('--train_csv', type=str, default=None, help='Path to the training data CSV file')
parser.add_argument('--val_csv', type=str, default=None, help='Path to the validation data CSV file')
# parser.add_argument('--test_csv', type=str, default=None, help='Optional: Path to a separate test data CSV file')


args = parser.parse_args()
# print("Parsed args:", args) 
# Load text prompts if path is provided (used by FOCUS)
if args.text_prompt_path:
    try:
        args.text_prompt = np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze()
        print(f"Loaded text prompts from {args.text_prompt_path}, shape: {args.text_prompt.shape}")
    except Exception as e:
        print(f"Error loading text prompts from {args.text_prompt_path}: {e}")
        args.text_prompt = None # Fallback or raise error
else:
    args.text_prompt = None # Ensure it's defined even if not provided

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

# Simplified settings dictionary
settings = {
    'task': args.task,
    'max_epochs': args.max_epochs,
    'results_dir': args.results_dir,
    'lr': args.lr,
    'experiment': args.exp_code,
    'seed': args.seed,
    'model_type': args.model_type,
    'mode': args.mode, # Retained as it's passed to Generic_MIL_Dataset
    "use_drop_out": args.drop_out,
    'weighted_sample': args.weighted_sample,
    'opt': args.opt,
    'reg': args.reg,
    'bag_loss': args.bag_loss,
    'train_csv': args.train_csv,
    'val_csv': args.val_csv,
    # 'test_csv': args.test_csv, # if you add it
    'data_folder_l': args.data_folder_l,
    'data_folder_s': args.data_folder_s,
    'text_prompt_path': args.text_prompt_path,
    'window_size': args.window_size,
    'sim_threshold': args.sim_threshold,
    'feature_dim': args.feature_dim,
    'max_context_length': args.max_context_length,
    'prototype_number': args.prototype_number
}
# Removed k-fold related settings: 'num_splits', 'k_start', 'k_end', 'label_frac' (as it's 1.0), 'split_dir'

print('\nLoad Dataset')

# Define n_classes and label_dict based on your specific task
# This needs to be set based on args.task before Generic_MIL_Dataset is called
if args.task == 'task_musk': # Example task name, use the one in your .sh script
    args.n_classes = 4 # For inflammation_level 0, 1, 2, 3
    specific_label_dict = {i: i for i in range(args.n_classes)}


elif args.task == 'task_another_custom': # Example for another task
    args.n_classes = 2
    specific_label_dict = {'class_A':0, 'class_B':1}
    # ... and so on for other tasks if you need them
else:
    # Fallback or error if task is not defined for direct loading
    # For now, assuming args.task passed from .sh will match one of the above,
    # or you will add a specific block for it.
    # If args.n_classes is directly set by .sh, this block might not be strictly needed to set n_classes
    # but specific_label_dict is still important.
    # If your .sh script already sets a default --task that defines n_classes (e.g. task_camelyon_subtyping),
    # and you override with your CSVs, ensure n_classes is correct.
    print(f"Warning: Task '{args.task}' not explicitly defined for n_classes/label_dict in main.py. " \
          f"Ensure --n_classes (if available in argparse) or default task n_classes is correct.")
    # It's safer to require args.task to be one that you explicitly define n_classes and label_dict for:
    if not hasattr(args, 'n_classes'): # If n_classes wasn't set by a predefined task block
        raise ValueError(f"args.n_classes not set. Please define it for task '{args.task}' or add a specific task block.")
    if 'specific_label_dict' not in locals():
        # Assuming labels in CSV are already 0, 1, ..., n_classes-1
        specific_label_dict = {str(i): i for i in range(args.n_classes)}


if not args.train_csv or not args.val_csv:
    raise ValueError("Please specify paths for --train_csv and --val_csv using command-line arguments.")

print(f"Loading training data from: {args.train_csv}")
train_dataset = Generic_MIL_Dataset(csv_path = args.train_csv,
                                    mode = args.mode,
                                    data_dir_s = args.data_folder_s,
                                    data_dir_l = args.data_folder_l,
                                    shuffle = True,
                                    print_info = True,
                                    label_dict = specific_label_dict,
                                    label_col = 'inflammation_label', # Your CSV label column name
                                    patient_strat= False) # Adjust if you use patient stratification

print(f"Loading validation data from: {args.val_csv}")
val_dataset = Generic_MIL_Dataset(csv_path = args.val_csv,
                                  mode = args.mode,
                                  data_dir_s = args.data_folder_s,
                                  data_dir_l = args.data_folder_l,
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = specific_label_dict,
                                  label_col = 'inflammation_label', # Your CSV label column name
                                  patient_strat= False)

# Using validation set as test set if no separate test_csv is provided
test_dataset = val_dataset
# if args.test_csv:
#     print(f"Loading test data from: {args.test_csv}")
#     test_dataset = Generic_MIL_Dataset(csv_path = args.test_csv, ...) # Provide full args

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

# Ensure results_dir is specific for this run if exp_code and seed are used
# (This was the original logic, seems okay)
args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

# The following 'split_dir' logic is no longer relevant for direct train/val
# print('split_dir: ', args.split_dir)
# assert os.path.isdir(args.split_dir)
# settings.update({'split_dir': args.split_dir}) # Remove or comment out


with open(os.path.join(args.results_dir, 'experiment_settings_{}.txt'.format(args.exp_code)), 'w') as f:
    for key, val in settings.items():
        print(f"{key}: {val}", file=f)
# f.close() # with open automatically closes

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))


def main(args):
    # train_dataset, val_dataset, test_dataset are now expected to be in the global scope
    # as defined above.

    current_run_id = 0 # Represents a single run, not a k-fold iteration
    
    seed_torch(args.seed) 
    
    datasets_tuple = (train_dataset, val_dataset, test_dataset)
    
    results, test_auc, val_auc, test_acc, val_acc, _, test_f1 = train(datasets_tuple, current_run_id, args)

    print(f"\n--- Training Run Results (ID {current_run_id}) ---")
    print(f"  Validation AUC: {val_auc:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Test Set AUC (evaluated on val_dataset): {test_auc:.4f}")
    print(f"  Test Set Accuracy (evaluated on val_dataset): {test_acc:.4f}")
    print(f"  Test Set F1-score (evaluated on val_dataset): {test_f1:.4f}")

    if not os.path.exists(args.results_dir): # Ensure again, though likely created above
        os.makedirs(args.results_dir)
    
    filename = os.path.join(args.results_dir, f'run_{current_run_id}_detailed_results.pkl')
    save_pkl(filename, results)

    summary_data = {
        'run_id': [current_run_id],
        'val_auc': [val_auc],
        'val_acc': [val_acc],
        'test_auc_on_val': [test_auc],
        'test_acc_on_val': [test_acc],
        'test_f1_on_val': [test_f1]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_filename = os.path.join(args.results_dir, 'summary_single_run.csv')
    summary_df.to_csv(summary_filename, index=False)

if __name__ == "__main__":
    # train_dataset, val_dataset, test_dataset are defined globally before main is called.
    main_results = main(args) # Assign to a variable if needed, though main() itself doesn't return 'results'
    print("finished!")
    print("end script")