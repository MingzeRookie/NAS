from __future__ import print_function
import argparse
import os
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset
import torch
import pandas as pd
import numpy as np

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
parser.add_argument('--data_folder_s', type=str, default=None, help='dir under data directory' )
parser.add_argument('--data_folder_l', type=str, default=None, help='dir under data directory' )
parser.add_argument('--max_epochs', type=int, default=200, help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None)
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--model_type', type=str, choices=['ViLa_MIL', 'FOCUS'], default='FOCUS', help='few-shot model for WSI classification')
parser.add_argument('--mode', type=str, choices=['transformer'], default='transformer')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'focal'], default='ce')
parser.add_argument('--task', type=str)
parser.add_argument("--text_prompt", type=str, default=None)
parser.add_argument("--text_prompt_path", type=str, default=None)
parser.add_argument("--prototype_number", type=int, default=16)
parser.add_argument("--window_size", type=int, default=8)
parser.add_argument("--sim_threshold", type=float, default=0.8)
parser.add_argument('--window_size', type=int, default=16, help='Window size for FOCUS model')
parser.add_argument('--sim_threshold', type=float, default=0.85, help='Similarity threshold for FOCUS model')
parser.add_argument('--feature_dim', type=int, default=1024, help='Dimension of input image features (x_l for FOCUS)')
parser.add_argument('--max_context_length', type=int, default=128, help='Max context length for FOCUS model token selection')

args = parser.parse_args()
args.text_prompt = np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze()

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

settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'label_frac': args.label_frac,
            'seed': args.seed,
            'model_type': args.model_type,
            'mode': args.mode,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

print('\nLoad Dataset')

if args.task == 'task_tcga_rcc_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = '/home/gzr/EasyMIL/dataset_csv/RCC.csv',
                                  mode = args.mode,
                                  data_dir_s = args.data_folder_s,
                                  data_dir_l = args.data_folder_l,
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'KICH':0, 'KIRC':1, 'KIRP':2},
                                  patient_strat= False,
                                  ignore=[])
    # #   data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
    # data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  
elif args.task == 'task_tcga_lung_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/home/gzr/EasyMIL/dataset_csv/LUAD_LUSC.csv',
                                  mode = args.mode,
                                  data_dir_s = args.data_folder_s,
                                  data_dir_l = args.data_folder_l,
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'LUAD':0, 'LUSC':1},
                                  patient_strat= False,
                                  ignore=[])

elif args.task == 'task_UBC-OCEAN_subtyping':
    args.n_classes=5
    dataset = Generic_MIL_Dataset(csv_path = '/home/gzr/EasyMIL/dataset_csv/UBC-OCEAN.csv',
                            mode = args.mode,
                            data_dir_s = args.data_folder_s,
                            data_dir_l = args.data_folder_l,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'CC': 0, 'HGSC': 1, 'LGSC': 2, 'EC': 3, 'MC': 4},
                            patient_strat=False,
                            ignore=[])
    
elif args.task == 'task_camelyon_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/home/gzr/EasyMIL/dataset_csv/camelyon.csv',
                            mode = args.mode,
                            data_dir_s = args.data_folder_s,
                            data_dir_l = args.data_folder_l,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal':0, 'tumor':1},
                            patient_strat=False,
                            ignore=[])
    
elif args.task == 'task_TUPAC16_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = '/home/gzr/EasyMIL/dataset_csv/TUPAC16.csv',
                            mode = args.mode,
                            data_dir_s = args.data_folder_s,
                            data_dir_l = args.data_folder_l,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_musk': # 使用您在 .sh 脚本中定义的 task 名称
    args.n_classes = 4 # 替换为您的实际类别数
    dataset = Generic_MIL_Dataset(
        csv_path = '/path/to/your/musk_dataset_main.csv', # 替换为您的主数据CSV文件路径
        mode = args.mode, # 通常是 'transformer'
        data_dir_s = args.data_folder_s, # 会从 .sh 脚本获取 (可能为 None 或空)
        data_dir_l = args.data_folder_l, # 会从 .sh 脚本获取 (指向 musk 特征)
        shuffle = False, # 或 True，根据您的需要
        print_info = True,
        label_dict = {'inflammation_level_0':0, 'inflammation_level_1':1, 'inflammation_level_2':2, 'inflammation_level_3':3}, # 替换为您的标签字典
        patient_strat= False, # 根据您的需要设置
        ignore=[]
    )

else:
    raise NotImplementedError

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))


def main(args):
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_f1 = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i)) 
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc, _, test_f1 = train(datasets, i, args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_f1.append(test_f1)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 'test_acc': all_test_acc, 'test_f1': all_test_f1})
    result_df = pd.DataFrame({'metric': ['mean', 'var'],
                              'test_auc': [np.mean(all_test_auc), np.std(all_test_auc)],
                              'test_f1': [np.mean(all_test_f1), np.std(all_test_f1)],
                              'test_acc': [np.mean(all_test_acc), np.std(all_test_acc)],
                              })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
        result_name = 'result_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
        result_name = 'result.csv'

    result_df.to_csv(os.path.join(args.results_dir, result_name), index=False)
    final_df.to_csv(os.path.join(args.results_dir, save_name))


if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


