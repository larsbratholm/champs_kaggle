import numpy as np
import torch

from cormorant.data.dataset_kaggle import KaggleTrainDataset
from cormorant.data.collate import collate_fn

datadir = '/Users/branderson/projects/cormorant/data/champs-scalar-coupling/'

data = np.load(datadir + 'targets_train.npz')
data = {key: val for key, val in data.items()}

num_data = len(data['charges'])
num_test = int(0.1*num_data)
num_valid = int(0.1*num_data)
num_train = num_data - num_test - num_valid

# Generate random permutation
np.random.seed(0)
data_perm = np.random.permutation(num_data)

# Now use the permutations to generate the indices of the dataset splits.
split_train, split_valid, split_test, split_extra = np.split(data_perm, [num_train, num_train+num_valid, num_train+num_valid+num_test])

assert(len(split_extra) == 0), 'Split was inexact {} {} {} {}'.format(len(split_train), len(split_valid), len(split_test), len(split_extra))

split_train = data_perm[split_train]
split_valid = data_perm[split_valid]
split_test = data_perm[split_test]

splits = {'train': split_train, 'valid': split_valid, 'test': split_test}

data_splits = {}
for split_name, split_idxs in splits.items():
    data_splits[split_name] = {}
    for key, val in data.items():
        data_splits[split_name][key] = val[split_idxs]

test_datasets = {split: KaggleTrainDataset(data) for split, data in data_splits.items()}

test_datasets['train'][50]

batch = [test_datasets['train'][idx] for idx in range(5)]

coll = collate_fn(batch, edge_features=['jj_1', 'jj_2', 'jj_3'])

breakpoint()
