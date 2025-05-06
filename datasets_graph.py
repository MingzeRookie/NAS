import argparse
import os
from glob import glob

import pandas as pd
import torch
from torch import Tensor
from torch_sparse import SparseTensor, cat
import torch_geometric
from torch_geometric.data import Data

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

        # yield train_loader, test_loader
        yield train_subset, test_subset

class NASHDataset(Dataset):
    def __init__(self, root_dir, d_list, frac='train', file_suffix='.pt'):
        self.root = root_dir
        self.bags_list = d_list
        self.frac = frac
        self.file_suffix = file_suffix
        self.bag_labels = []
        self.bag_feats_list = []
        for bag_idx, label in tqdm(self.bags_list):
            bag_dir = os.path.join(self.root, bag_idx + self.file_suffix)
            # bag_feats = pd.read_csv(bag_dir, index_col=0)
            # bag_feats = bag_feats.interpolate(method="linear").to_numpy()
            # bag_feats = torch.tensor(bag_feats,dtype=torch.float32)
            # bag_feats = torch.unique(bag_feats, dim=0)
            bag_graph = torch.load(bag_dir, weights_only=False)
            bag_label = torch.tensor([label], dtype=torch.float32)
            self.bag_feats_list.append(bag_graph)
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
        # bag = BatchWSI.from_data_list([bag_feats], update_cat_dims={'edge_latent': 1})
        bag_label = self.bag_labels[i]
        nan_mask = torch.isnan(bag_feats.x)
        bag_feats.x[nan_mask] = 0.0
        bag_feats.x = torch.unique(bag_feats.x, dim=0)
        return bag_feats, bag_label 

class BatchWSI(torch_geometric.data.Batch):
    def __init__(self):
        super(BatchWSI, self).__init__()
        pass
    
    @classmethod
    def from_data_list(cls, data_list, follow_batch=[], exclude_keys=[], update_cat_dims={}):
            r"""Constructs a batch object from a python list holding
            :class:`torch_geometric.data.Data` objects.
            The assignment vector :obj:`batch` is created on the fly.
            Additionally, creates assignment batch vectors for each key in
            :obj:`follow_batch`.
            Will exclude any keys given in :obj:`exclude_keys`."""

            keys = list(set(data_list[0].keys()) - set(exclude_keys))
            assert 'batch' not in keys and 'ptr' not in keys

            batch = cls()
            for key in data_list[0].__dict__.keys():
                if key[:2] != '__' and key[-2:] != '__':
                    batch[key] = None

            batch.__num_graphs__ = len(data_list)
            batch.__data_class__ = data_list[0].__class__
            for key in keys + ['batch']:
                batch[key] = []
            batch['ptr'] = [0]
            cat_dims = {}
            device = None
            slices = {key: [0] for key in keys}
            cumsum = {key: [0] for key in keys}
            num_nodes_list = []
            for i, data in enumerate(data_list):
                for key in keys:
                    item = data[key]

                    # Increase values by `cumsum` value.
                    cum = cumsum[key][-1]
                    if isinstance(item, Tensor) and item.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            item = item + cum
                    elif isinstance(item, SparseTensor):
                        value = item.storage.value()
                        if value is not None and value.dtype != torch.bool:
                            if not isinstance(cum, int) or cum != 0:
                                value = value + cum
                            item = item.set_value(value, layout='coo')
                    elif isinstance(item, (int, float)):
                        item = item + cum

                    # Gather the size of the `cat` dimension.
                    size = 1
                    
                    if key in update_cat_dims.keys():
                        cat_dim = update_cat_dims[key]
                    else:
                        cat_dim = data.__cat_dim__(key, data[key])
                        # 0-dimensional tensors have no dimension along which to
                        # concatenate, so we set `cat_dim` to `None`.
                        if isinstance(item, Tensor) and item.dim() == 0:
                            cat_dim = None
                    
                    cat_dims[key] = cat_dim

                    # Add a batch dimension to items whose `cat_dim` is `None`:
                    if isinstance(item, Tensor) and cat_dim is None:
                        cat_dim = 0  # Concatenate along this new batch dimension.
                        item = item.unsqueeze(0)
                        device = item.device
                    elif isinstance(item, Tensor):
                        size = item.size(cat_dim)
                        device = item.device
                    elif isinstance(item, SparseTensor):
                        size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]
                        device = item.device()

                    batch[key].append(item)  # Append item to the attribute list.

                    slices[key].append(size + slices[key][-1])
                    inc = data.__inc__(key, item)
                    if isinstance(inc, (tuple, list)):
                        inc = torch.tensor(inc)
                    cumsum[key].append(inc + cumsum[key][-1])

                    if key in follow_batch:
                        if isinstance(size, Tensor):
                            for j, size in enumerate(size.tolist()):
                                tmp = f'{key}_{j}_batch'
                                batch[tmp] = [] if i == 0 else batch[tmp]
                                batch[tmp].append(
                                    torch.full((size, ), i, dtype=torch.long,
                                               device=device))
                        else:
                            tmp = f'{key}_batch'
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size, ), i, dtype=torch.long,
                                           device=device))

                if hasattr(data, '__num_nodes__'):
                    num_nodes_list.append(data.__num_nodes__)
                else:
                    num_nodes_list.append(None)

                num_nodes = data.num_nodes
                if num_nodes is not None:
                    item = torch.full((num_nodes, ), i, dtype=torch.long,
                                      device=device)
                    batch.batch.append(item)
                    batch.ptr.append(batch.ptr[-1] + num_nodes)

            batch.batch = None if len(batch.batch) == 0 else batch.batch
            batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
            batch.__slices__ = slices
            batch.__cumsum__ = cumsum
            batch.__cat_dims__ = cat_dims
            batch.__num_nodes_list__ = num_nodes_list

            ref_data = data_list[0]
            for key in batch.keys():
                items = batch[key]
                item = items[0]
                
                ### <--- Updating Cat Dim
                if key in update_cat_dims.keys():
                    cat_dim = update_cat_dims[key]
                else: 
                    cat_dim = ref_data.__cat_dim__(key, item)
                    cat_dim = 0 if cat_dim is None else cat_dim
                ### ---?
                if isinstance(item, Tensor):
                    batch[key] = torch.cat(items, cat_dim)
                elif isinstance(item, SparseTensor):
                    batch[key] = cat(items, cat_dim)
                elif isinstance(item, (int, float)):
                    batch[key] = torch.tensor(items)

            if torch_geometric.is_debug_enabled():
                batch.debug()

            return batch.contiguous()
        
if __name__ == '__main__':
    pass
