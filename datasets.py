import argparse
import os
from glob import glob

import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def cross_validation(cfg):
    '''

    '''
    # Dataset setting
    dataset_cfg = cfg['dataset']
    fold = dataset_cfg.get('k_fold', 5)
    root_dir = dataset_cfg.get('root_dir','datasets/UNI-feature')
    labels_dir = dataset_cfg.get('label_dir','datasets/labels.csv')
    task = dataset_cfg.get('target', 'steatosis')
    batch_size = dataset_cfg.get('bs', 1)
    feats_suffix = dataset_cfg.get('file_suffix', '.csv') # need the '.'
    random_seed = dataset_cfg.get('random', 7777)

    # k-fold initial
    kfold = KFold(n_splits=fold, shuffle=True, random_state=random_seed)

    # loading
    label_file = pd.read_csv(labels_dir,index_col=0)
    label_file.index = label_file.index.map(str)
    feats_dir = os.listdir(root_dir)
    bags_list = [str(i) for i in label_file.index if str(i) + feats_suffix in feats_dir]
    labels = [label_file.loc[i,task] for i in bags_list]
    stage_map = {'low':0,'boardline':1,'high':2}
    if task == 'NAS-stage':
        labels = [stage_map[i] for i in labels]
    data_map_list = list(zip(bags_list,labels))

    for fold, (train_indices, test_indices) in enumerate(kfold.split(data_map_list)):
        # print(f"Fold {fold + 1}")
        train_list = [data_map_list[i] for i in train_indices]
        test_list = [data_map_list[i] for i in test_indices]
        train_subset = NASHDataset(root_dir, train_list, 'train', file_suffix=feats_suffix)
        test_subset = NASHDataset(root_dir, test_list, 'test', file_suffix=feats_suffix)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,num_workers=64)
        test_loader = DataLoader(
            test_subset, batch_size=batch_size, shuffle=False, num_workers=16
        )

        yield train_loader, test_loader

class NASHDataset(Dataset):
    def __init__(self, root_dir, d_list, frac='train', file_suffix='.csv'):
        self.root = root_dir
        self.bags_list = d_list
        self.frac = frac
        self.file_suffix = file_suffix
        self.bag_feats_list = []
        self.bag_labels = []
        for bag_idx, label in tqdm(self.bags_list):
            bag_dir = os.path.join(self.root, bag_idx + self.file_suffix)
            bag_feats = torch.load(bag_dir)
            # bag_feats = bag_feats['bag_feats'].type(torch.float32)
            bag_feats = torch.stack([bag_feats[i] for i in bag_feats],dim=0).type(torch.float32)
            # bag_feats = torch.tensor(bag_feats,dtype=torch.float32)
            bag_label = torch.tensor(label, dtype=torch.float32)
            self.bag_feats_list.append(bag_feats)
            self.bag_labels.append(bag_label)

    def __len__(self):

        return len(self.bags_list)

    def __getitem__(self, i):
        # bag_idx, label = self.bags_list[i]
        # bag_dir = os.path.join(self.root, bag_idx + self.file_suffix)
        # bag_feats = pd.read_csv(bag_dir, index_col=0)
        # bag_feats = bag_feats.interpolate(method="linear").to_numpy()
        # bag_feats = torch.tensor(bag_feats,dtype=torch.float32)
        # bag_feats = torch.unique(bag_feats, dim=0)
        # # if self.frac == 'train':
        # #     pass
        # bag_label = torch.tensor(label,dtype=torch.float32)
        bag_feats = self.bag_feats_list[i]
        bag_label = self.bag_labels[i]

        return bag_feats, bag_label

if __name__ == '__main__':
    pass
