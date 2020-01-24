#!/usr/bin/env python
# coding: utf-8

# # Setup

import numpy as np
import pandas as pd


def normalize_scalar(scalar_traj, stats=None):
    flat_traj = np.hstack(scalar_traj)
    if stats is None:
        mean = np.mean(flat_traj)
        std = np.std(flat_traj)
    else:
        mean, std = stats
    scalar_traj = np.array([(ci - mean) / std for ci in scalar_traj])
    return scalar_traj, mean, std


datadir = './data/champs-scalar-coupling/'

# ## Load main data in to DataFrame
data_configurations = pd.read_csv(datadir+'structures.csv', index_col=[0, 1])

data_train = pd.read_csv(datadir+'train.csv')
data_test = pd.read_csv(datadir+'test.csv')

shieldings = pd.read_csv(datadir + 'magnetic_shielding_tensors.csv')
coupling_contribs = pd.read_csv(datadir + 'scalar_coupling_contributions.csv')
mulliken_charges = pd.read_csv(datadir + 'mulliken_charges.csv')


atom_charges = np.zeros(len(data_configurations['atom'].values), dtype=np.int)
for t, chrg in {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}.items():
    atom_charges[data_configurations['atom'].values == t] = chrg

data_configurations['charges'] = atom_charges

num_atoms = data_configurations.reset_index().groupby('molecule_name')['atom_index'].nunique()

atom_positions = np.split(data_configurations[['x', 'y', 'z']].values, (np.cumsum(num_atoms.values)))[:-1]
atom_types = np.split(data_configurations['atom'].values, (np.cumsum(num_atoms.values)))[:-1]


# Keep old type string
data_train['type_str'] = data_train['type']
data_test['type_str'] = data_test['type']

# For now, convert type string to int 1, 2, 3
data_train['type'] = data_train['type'].apply(lambda s: int(s[0]))
data_test['type'] = data_test['type'].apply(lambda s: int(s[0]))


# ### Get the molecules in the training and test sets

train_molecules = data_train['molecule_name'].unique()
test_molecules = data_test['molecule_name'].unique()
data_configs_reset = data_configurations.reset_index()
train_configs = data_configs_reset['molecule_name'].isin(train_molecules)

data_configs_train = data_configs_reset.loc[data_configs_reset['molecule_name'].isin(train_molecules)]
test_configs = data_configs_reset['molecule_name'].isin(test_molecules)
data_configs_test = data_configs_reset.loc[data_configs_reset['molecule_name'].isin(test_molecules)]

data_configs_train = pd.DataFrame.copy(data_configs_train)
mc_charges, mean_partial_qs, std_partial_qs = normalize_scalar(mulliken_charges['mulliken_charge'])
data_configs_train['partial_qs'] = mc_charges

num_atoms_train = data_configs_train.groupby('molecule_name')['atom_index'].nunique()
num_atoms_test = data_configs_test.groupby('molecule_name')['atom_index'].nunique()

atom_charges_train = np.split(data_configs_train['charges'].values, (np.cumsum(num_atoms_train.values)))[:-1]
atom_partial_qs_train = np.split(data_configs_train['partial_qs'].values, (np.cumsum(num_atoms_train.values)))[:-1]
atom_positions_train = np.split(data_configs_train[['x', 'y', 'z']].values, (np.cumsum(num_atoms_train.values)))[:-1]

atom_charges_test = np.split(data_configs_test['charges'].values, (np.cumsum(num_atoms_test.values)))[:-1]
atom_positions_test = np.split(data_configs_test[['x', 'y', 'z']].values, (np.cumsum(num_atoms_test.values)))[:-1]

split_train = np.split(data_train.values, np.cumsum(data_train.groupby('molecule_name')['id'].nunique().values))
split_test = np.split(data_test.values, np.cumsum(data_test.groupby('molecule_name')['id'].nunique().values))

coupling_contribs_split_train = np.split(coupling_contribs.values, np.cumsum(data_train.groupby('molecule_name')['id'].nunique().values))

train_mol_idxs = np.array([idx[-6:] for idx in train_molecules]).astype('int') - 1
test_mol_idxs = np.array([idx[-6:] for idx in test_molecules]).astype('int') - 1


train_zeros = np.array([np.zeros(ci.shape) for ci in atom_charges_train])
test_zeros = np.array([np.zeros(ci.shape) for ci in atom_charges_test])
print(train_zeros[0].shape, train_zeros[1].shape)
print(atom_partial_qs_train[0].shape, atom_partial_qs_train[1].shape)
split_train = split_train[:-1]
split_test = split_test[:-1]

# ### Create basic test dataset and save to file
targets_train = {
    'jj_edge': np.array([split_train[idx][:, [2, 3]].astype(np.int) for idx in range(len(split_train))]),
    'jj_type': np.array([split_train[idx][:, 4].astype(np.int) for idx in range(len(split_train))]),
    'jj_value': np.array([split_train[idx][:, 5].astype(np.double) for idx in range(len(split_train))]),
    'jj_label': np.array([split_train[idx][:, 6].astype(str) for idx in range(len(split_train))])
}


# ### Create basic test dataset and save to file
targets_test = {
    'jj_edge': np.array([split_test[idx][:, [2, 3]].astype(np.int) for idx in range(len(split_test))]),
    'jj_type': np.array([split_test[idx][:, 4].astype(np.int) for idx in range(len(split_test))]),
    'jj_value': np.array([np.nan * np.empty(split_test[idx][:, 4].shape).astype('float') for idx in range(len(split_test))]),
    'jj_label': np.array([split_test[idx][:, 5].astype(str) for idx in range(len(split_test))])
}

config_train = {'index': np.array(train_molecules),
                'charges': np.array(atom_charges_train),
                'positions': np.array(atom_positions_train),
                'partial_qs': np.array(atom_partial_qs_train),
                'zeros': train_zeros,
                }


config_test = {'index': np.array(test_molecules),
               'charges': np.array(atom_charges_test),
               'positions': np.array(atom_positions_test),
               'partial_qs': test_zeros,
               'zeros': test_zeros,
               }


data_targets_train = {**config_train, **targets_train}
data_targets_test = {**config_test, **targets_test}
stats = {'mean_partial_qs': mean_partial_qs,
         'std_partial_qs': std_partial_qs
         }

nmolecules_train = data_targets_train['charges'].shape[0]
test_choice = np.random.choice(nmolecules_train, 5000)
test_choice.sort()

data_targets_dummy_train = {key: val[test_choice] for (key, val) in data_targets_train.items()}
print('sorted type!')


np.savez_compressed(datadir + 'targets_train_expanded.npz', **data_targets_train)
np.savez_compressed(datadir + 'targets_train_dummy_sub.npz', **data_targets_dummy_train)
np.savez_compressed(datadir + 'targets_train_expanded_stats.npz', **stats)
np.savez_compressed(datadir + 'targets_test_expanded.npz', **data_targets_test)
