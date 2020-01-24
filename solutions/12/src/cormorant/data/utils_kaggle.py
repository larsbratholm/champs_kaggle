import torch
import numpy as np

import logging
import os

from torch.utils.data import DataLoader
from cormorant.data.dataset_kaggle import KaggleTrainDataset
from cormorant.data.prepare import prepare_dataset


def init_nmr_kaggle_dataset(args, datadir, file_name='targets_train.npz', trim=True, additional_atom_features=None, use_all_train=False):
    data = np.load(datadir + 'champs-scalar-coupling/' + file_name, allow_pickle=True)
    data = {key: val for (key, val) in data.items()}
    
    num_data = len(data['charges'])
    num_test = int(0.1*num_data)
    num_valid = int(0.1*num_data)
    num_train = num_data - num_test - num_valid

    # Generate random permutation
    np.random.seed(0)
    data_perm = np.random.permutation(num_data)[:(num_train + num_valid + num_test)]

    # Now use the permutations to generate the indices of the dataset splits.
    split_train, split_valid, split_test, split_extra = np.split(data_perm, [num_train, num_train+num_valid, num_train+num_valid+num_test])

    assert(len(split_extra) == 0), 'Split was inexact {} {} {} {}'.format(len(split_train), len(split_valid), len(split_test), len(split_extra))

    split_train = data_perm[split_train]
    split_valid = data_perm[split_valid]
    split_test = data_perm[split_test]

    if args.num_train > 0:
        split_train = split_train[:args.num_train]

    if args.num_valid > 0:
        split_valid = split_valid[:args.num_valid]

    if args.num_test > 0:
        split_test = split_test[:args.num_test]
    
    if use_all_train:
        split_train = data_perm

    splits = {'train': split_train, 'valid': split_valid, 'test': split_test}

    data_splits = {}
    for split_name, split_idxs in splits.items():
        data_splits[split_name] = {}
        for key, val in data.items():
            data_splits[split_name][key] = val[split_idxs]

    datasets = {split: KaggleTrainDataset(data, additional_atom_features=additional_atom_features) for split, data in data_splits.items()}
    
    if trim:
        datasets = {split: trim_dataset(args.target, dataset) for split, dataset in datasets.items()}


    # Now, update the number of training/test/validation sets in args
    args.num_train = datasets['train'].num_pts
    args.num_valid = datasets['valid'].num_pts
    args.num_test = datasets['test'].num_pts

    # These parameters are necessary to initialize the network
    num_species = datasets['train'].num_species
    max_charge = datasets['train'].max_charge

    return args, datasets, num_species, max_charge


def init_nmr_eval_kaggle_dataset(args, datadir, file_name='targets_test.npz', trim=True, additional_atom_features=None):
    data = np.load(datadir + 'champs-scalar-coupling/' + file_name, allow_pickle=True)
    data = {key: val for (key, val) in data.items()}
    datasets = {'test': KaggleTrainDataset(data, additional_atom_features=additional_atom_features, shuffle=False)}
    
    if trim:
        datasets = {split: trim_dataset(args.target, dataset) for split, dataset in datasets.items()}

    args.num_test = datasets['test'].num_pts

    # These parameters are necessary to initialize the network
    num_species = datasets['test'].num_species
    max_charge = datasets['test'].max_charge

    return args, datasets, num_species, max_charge


def trim_dataset(target, dataset):
    """
    Removes entries from the dataset that do not have values for the target.
    Note that this changes the dataset in place.

    Parameters
    ----------
    target : str
        Name of the target being optimized for
    dataset : KaggleTrainDataset object
        Dataset to be trimmed

    Returns
    -------
    data : KaggleTrainDataset
        Trimmed dataset.
    """
    target_column = dataset.data[target + "_value"]
    nonempty_locs = np.where(np.array([len(val) for val in target_column]) > 0)[0]

    for key, attrib in dataset.data.items():
        new_attrib = attrib[nonempty_locs]
        dataset.data[key] = new_attrib
    dataset.num_pts = len(nonempty_locs)
    dataset._build_perm()
    # print(dataset.perm)
    return dataset
