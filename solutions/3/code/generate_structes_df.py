#!/usr/bin/env python
# coding: utf-8

import os
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from multiprocessing import Pool


def seed_everything(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


# Get script location
script_dir = os.path.abspath(os.path.dirname(__file__))

train_df = pd.read_csv(script_dir + '/../../data/train.csv')
test_df = pd.read_csv(script_dir + '/../../data/test.csv')
structures_df = pd.read_csv(script_dir + '/../../data/structures.csv')
test_df['scalar_coupling_constant'] = np.nan


# # 추출하기

mol_names = list(structures_df['molecule_name'].unique())


def read_mol(mol_name):
    QM9_PATH = './input/quantum-machine-9-aka-qm9'
    path = os.path.join(QM9_PATH, mol_name+'.xyz')

    stats = pd.read_csv(path,sep=' |\t',engine='python',skiprows=1,nrows=1,header=None)
    stats = stats.loc[:,2:]
    stats.columns = ['rc_A','rc_B','rc_C','mu','alpha','homo','lumo','gap','r2','zpve','U0','U','H','G','Cv']

    mu = pd.read_csv(path,sep='\t',engine='python', skiprows=2, 
                     skipfooter=3, names=['atom', 'x', 'y', 'z', 'mu'])
    return mol_name, stats, mu['mu']



dipole_moments = {}
mol_metas = {}
n_cpu = 20
with Pool(n_cpu) as p:
    n = len(mol_names)
    with tqdm(total=n) as pbar:
        for res in p.imap_unordered(read_mol, mol_names):
            mol_metas[res[0]] = res[1]
            dipole_moments[res[0]] = res[2]
            pbar.update()
dipole_moments = pd.concat([dipole_moments[mol_name] for mol_name in mol_names])
dipole_moments = dipole_moments.reset_index(drop=True)
dipole_moments = dipole_moments.replace(to_replace=r'\*\^', value='e', regex=True).astype('float64')

mol_metas = pd.concat([mol_metas[mol_name] for mol_name in mol_names])
mol_metas = mol_metas.reset_index(drop=True)
mol_metas = mol_metas.replace(to_replace=r'\*\^', value='e', regex=True).astype('float64')


# In[182]:


mol_metas['molecule_name'] = mol_names
mol_metas = mol_metas[['molecule_name'] + list(mol_metas.columns[:-1])]
structures_df['mu'] = dipole_moments


# In[143]:


structures_df.to_csv('./input/champs-scalar-coupling/structures_mu.csv', index=False)


