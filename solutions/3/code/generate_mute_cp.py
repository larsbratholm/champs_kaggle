#!/usr/bin/env python
# coding: utf-8

import os
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


def seed_everything(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


script_dir = os.path.abspath(os.path.dirname(__file__))

train_df = pd.read_csv(script_dir + '/../../data/train.csv')
test_df = pd.read_csv(script_dir + '/../../data/test.csv')
structures_df = pd.read_csv(script_dir + '/../../data/structures.csv')
test_df['scalar_coupling_constant'] = np.nan


# # is_coupled

both_df = pd.concat([train_df, test_df])
both_df = both_df.sort_values('molecule_name').reset_index(drop=True)


merged_df = pd.concat([both_df[['molecule_name', 'atom_index_0']].rename({'atom_index_0':'atom_index'}, axis=1), 
                       both_df[['molecule_name', 'atom_index_1']].rename({'atom_index_1':'atom_index'}, axis=1)], axis=0)
merged_df = merged_df.sort_values('molecule_name').reset_index(drop=True)
merged_df = merged_df.drop_duplicates()
merged_df['is_coupled'] = 1


structures_extra_df = structures_df.merge(merged_df, how='left', left_on=['molecule_name', 'atom_index'], 
                                right_on=['molecule_name', 'atom_index']
                               )
structures_extra_df = structures_extra_df.fillna(0)


# # Add column of nearest_atom_index


mol_names = structures_extra_df['molecule_name'].unique()
structures_extra_group = structures_extra_df.groupby('molecule_name')


def add_nearest_atom(mol_df):
    mol_pos = mol_df[['x', 'y', 'z']].values    
    nbrs = NearestNeighbors(n_neighbors=len(mol_pos), algorithm='ball_tree').fit(mol_pos)
    distances, indices = nbrs.kneighbors(mol_pos)    
    return indices[:, 1]
nearest_indices = []
for mol_name in tqdm(mol_names):
    nearest_indices.append(add_nearest_atom(structures_extra_group.get_group(mol_name)))
nearest_indices = np.concatenate(nearest_indices)
structures_extra_df['nearest_atom_index'] = nearest_indices


structures_nn_df = structures_extra_df.merge(structures_extra_df[['molecule_name', 'atom_index', 'atom']], 
                          how='left', left_on=['molecule_name', 'nearest_atom_index'],
                         right_on=['molecule_name', 'atom_index'], suffixes=('_1', '_0'))
structures_nn_df = structures_nn_df.drop('atom_index_0', axis=1)
structures_nn_df = structures_nn_df.rename({'nearest_atom_index':'atom_index_0'}, axis=1)
structures_nn_df['type'] = '1J' + structures_nn_df['atom_0'].str.cat(structures_nn_df['atom_1'])


both_df = structures_nn_df[structures_nn_df['is_coupled']==0][['molecule_name', 'atom_index_0', 'atom_index_1', 'type']]
both_df['scalar_coupling_constant'] = 1e08 # if this value is included in calculation of loss, the loss became larger.
both_df['id'] = -1


gened_train_df = both_df[both_df['molecule_name'].isin(train_df['molecule_name'].unique())]
gened_test_df = both_df[both_df['molecule_name'].isin(test_df['molecule_name'].unique())]


new_train_df = pd.concat([gened_train_df[train_df.columns], train_df]).sort_values('molecule_name').reset_index(drop=True)
new_test_df = pd.concat([gened_test_df[test_df.columns], test_df]).sort_values('molecule_name').reset_index(drop=True)


new_train_df.to_csv('./input/champs-scalar-coupling/train_mute_cp.csv', index=False)


# In[43]:


new_test_df.to_csv('./input/champs-scalar-coupling/test_mute_cp.csv', index=False)

