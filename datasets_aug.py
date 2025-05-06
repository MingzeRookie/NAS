import argparse
import os
from glob import glob

import pandas as pd
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import numpy as np

def cross_validation(cfg):
    """ """
    # Dataset setting
    dataset_cfg = cfg["dataset"]
    fold = dataset_cfg.get("k_fold", 5)
    root_dir = dataset_cfg.get("root_dir", "datasets/UNI-feature")
    labels_dir = dataset_cfg.get("label_dir", "datasets/labels.csv")
    task = dataset_cfg.get("target", "steatosis")
    batch_size = dataset_cfg.get("bs", 1)
    feats_suffix = dataset_cfg.get("file_suffix", ".csv")  # need the '.'
    random_seed = dataset_cfg.get("random", 7777)
    cluster_size = dataset_cfg.get("cluster_size",32)
    # k-fold initial
    kfold = KFold(n_splits=fold, shuffle=True, random_state=random_seed)

    # loading
    label_file = pd.read_csv(labels_dir, index_col=0)
    label_file.index = label_file.index.map(str)
    feats_dir = os.listdir(root_dir)
    bags_list = [str(i) for i in label_file.index if str(i) + feats_suffix in feats_dir]
    labels = [label_file.loc[i, task] for i in bags_list]
    data_map_list = list(zip(bags_list, labels))

    for fold, (train_indices, test_indices) in enumerate(kfold.split(data_map_list)):
        # print(f"Fold {fold + 1}")
        train_list = [data_map_list[i] for i in train_indices]
        test_list = [data_map_list[i] for i in test_indices]
        train_subset = NASHDataset(
            root_dir, train_list, "train", cluster_size, file_suffix=feats_suffix
        )
        test_subset = NASHDataset(root_dir, test_list, "test", file_suffix=feats_suffix)
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=16
        )
        test_loader = DataLoader(
            test_subset, batch_size=batch_size, shuffle=False, num_workers=16
        )

        yield train_loader, test_loader


class NASHDataset(Dataset):

    def __init__(
        self, root_dir, d_list, frac="train",  n_clusters=32, file_suffix=".csv"
    ):
        self.root = root_dir
        self.bags_list = d_list
        self.frac = frac
        self.file_suffix = file_suffix
        self.n_clusters = n_clusters
        self.cluster_method = MiniBatchKMeans(n_clusters=n_clusters)
        self.bag_feats_list = []
        self.bag_labels = []
        for bag_idx, label in tqdm(self.bags_list):
            bag_dir = os.path.join(self.root, bag_idx + self.file_suffix)
            bag_feats = pd.read_csv(bag_dir, index_col=0)
            bag_feats = bag_feats.interpolate(method="linear")
            bag_feats = np.unique(bag_feats.to_numpy(), axis=0)
            self.bag_feats_list.append(bag_feats)
            self.bag_labels.append(label)

    def __len__(self):

        return len(self.bags_list)

    def __getitem__(self, i):
        bag_feats = self.bag_feats_list[i]
        bag_label = self.bag_labels[i]

        if self.frac == "train" and np.random.uniform() > 0.5:
            bag_feats = self.cluster_sampling(bag_feats)
        bag_feats = torch.tensor(bag_feats, dtype=torch.float32)
        bag_label = torch.tensor(bag_label, dtype=torch.float32)
        return bag_feats, bag_label

    def cluster_sampling(self, sample):
        self.cluster_method.fit(sample)
        cluster_labels = self.cluster_method.labels_
        ratio = np.random.randint(1, 100, 1) * 0.01
        new_sample = []
        for c in range(self.n_clusters):
            cluster_sample = sample[cluster_labels == c]
            sample_n = int(np.ceil(cluster_sample.shape[0]*ratio))
            slice = np.random.choice(range(len(cluster_sample)), sample_n)
            new_sample.extend(cluster_sample[slice])
        return np.array(new_sample)


if __name__ == "__main__":
    pass
