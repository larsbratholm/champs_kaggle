import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

type2idx = {'3JHC':0, '2JHC':1, '1JHC':2, '3JHH':3, '2JHH':4, '3JHN':5, '2JHN':6, '1JHN':7,
            '1JHO':8, '1JCO':9, '1JOC':9, '1JCN':10, '1JNC':10, '1JNO':11, '1JON':11, '1JCC':12, '1JNN':13, 
             '1JCF':14, '1JFC':14}
atom2idx = {'C':0, 'H':1, 'N':2, 'O':3, 'F':4}


class MolDB(Dataset):
    def __init__(self, df, max_ncouplings):
        self.df = df
        self.mol_indices = list(df.groupby('molecule_name').groups.values())
        self.type = df['type'].map(type2idx).values
        self.max_ncouplings = max_ncouplings        
        
    def __len__(self):
        return len(self.mol_indices)  
    
    def __getitem__(self, idx):
        indices = self.mol_indices[idx]
        rows = self.df.iloc[indices]        
        natomes = len(rows)
        
        atom0 = torch.LongTensor(self.max_ncouplings).zero_()
        atom1 = torch.LongTensor(self.max_ncouplings).zero_()
        atom0[:natomes] = torch.LongTensor(rows['atom_0'].map(atom2idx).values)
        atom1[:natomes] = torch.LongTensor(rows['atom_1'].map(atom2idx).values)
        
        typ = torch.LongTensor(self.max_ncouplings).zero_()
        typ[:natomes] = torch.LongTensor(self.type[indices])       
        
        xyz0 = torch.FloatTensor(self.max_ncouplings, 3).zero_()        
        xyz0[:natomes, :] = torch.FloatTensor(rows[['x_0','y_0','z_0']].values)
        xyz1 = torch.FloatTensor(self.max_ncouplings, 3).zero_()
        xyz1[:natomes, :] = torch.FloatTensor(rows[['x_1','y_1','z_1']].values)
        
        mu0 = torch.FloatTensor(self.max_ncouplings, 1).zero_()
        mu0[:natomes, :] = torch.FloatTensor(rows[['mu_0']].values)
        mu1 = torch.FloatTensor(self.max_ncouplings, 1).zero_()
        mu1[:natomes, :] = torch.FloatTensor(rows[['mu_1']].values)       
        
        # not currently used but reserved.
        weight = torch.FloatTensor(self.max_ncouplings).zero_()
        weight[:natomes] = torch.FloatTensor(rows['weight'].values) 
        
        target = torch.FloatTensor(self.max_ncouplings, 5).zero_()
        target[:natomes, :] = torch.FloatTensor(rows[['scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso']].values)
        
        mask = torch.ByteTensor(self.max_ncouplings).zero_()
        mask[:natomes] = 1
        
        return atom0, atom1, typ, xyz0, xyz1, mu0, mu1, mask, target, weight, natomes


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis) + 1e-8
    axis = axis / (math.sqrt(np.dot(axis, axis)))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


class MolDB_FromDB(Dataset):
    def __init__(self, db, times=1):
        loader = DataLoader(
            db, batch_size=64, shuffle=False,
            num_workers=16)
        bunches = [[] for _ in range(11) ]
        for i, batch in enumerate(tqdm(loader)):
            for j in range(len(batch)):                
                bunches[j].append(batch[j].clone())
        bunches = [torch.cat(bunch) for bunch in bunches]
                
        self.bunches = bunches
        self.times = times
        self.nsamples = len(self.bunches[-1])        
    
    def __getitem__(self, idx):
        idx = idx % self.nsamples        
        trans_noise = np.random.normal(0, 2, 3)
        rot_noise = torch.FloatTensor(
            rotation_matrix( trans_noise, np.random.normal(0, 3.14/2) )).transpose(0, 1)
        trans_noise = torch.FloatTensor(trans_noise)            
        
        bunches = [bunch[idx] for bunch in self.bunches]
        
        xyz0 = bunches[3]
        xyz1 = bunches[4]
        xyz0 = torch.matmul(xyz0, rot_noise) + trans_noise
        xyz1 = torch.matmul(xyz1, rot_noise) + trans_noise        
        bunches[3] = xyz0
        bunches[4] = xyz1
        
        return bunches
        
    def __len__(self):
        return self.nsamples*self.times


def main():
    db = MolDB(pd.read_csv('../../input/custom/train.csv'), 135)     
    db = MolDB_FromDB(db)
    
    #samples = [(*sample) for sample in samples]
        


if __name__ == '__main__':
    main()