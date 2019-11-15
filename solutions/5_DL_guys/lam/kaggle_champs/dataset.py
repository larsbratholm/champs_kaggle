import os
import numpy as np
import openbabel
import torch

from torch_geometric.data import Data
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
from . import constants


class MoleculeDataset(Dataset):
    def __init__(self, metadata=None, base_dir=None, transform=None, feat_v2=False):
        self.metadata = metadata
        self.base_dir = base_dir
        self.transform = transform
        self.conversion = openbabel.OBConversion()
        self.conversion.SetInAndOutFormats("xyz", "mdl")
        self.feat_v2 = feat_v2

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        mol = openbabel.OBMol()
        mol_name = self.metadata.iloc[index]['molecule_name']

        xyz_file = os.path.join(self.base_dir, f'{mol_name}.xyz')
        if not os.path.exists(xyz_file):
            raise FileNotFoundError(f'Expecting file {xyz_file} not found')
        self.conversion.ReadFile(mol, xyz_file)

        if self.feat_v2:
            data = mol_to_data_v2(mol)
        else:
            data = mol_to_data(mol)
        data = self._add_targets(index, data)
        if self.transform:
            data = self.transform(data)
        return data

    def _add_targets(self, index, data):
        return data
   
class FP16_Data(Data):
    
    def to(self, device, *keys):
        r"""Performs tensor dtype and/or device conversion to all attributes
        :obj:`*keys`.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        keys = [key for key, item in self(*keys) if self[key].dtype != torch.long]
        return self.apply(lambda x: x.to(device), *keys)

    
class ChampsDataset(MoleculeDataset):
    def __init__(self,
                 metadata=None,
                 base_dir=None,
                 transform=None,
                 feat_v2=False,
                 alternate_direction=False):
        super(ChampsDataset, self).__init__(metadata=metadata,
                                         base_dir=base_dir,
                                         transform=transform,
                                         feat_v2=feat_v2)
        self.metadata = metadata.loc[:, ['molecule_name']].drop_duplicates()
        self.source_data = dict([
            (ind, df) for ind, df in tqdm(metadata.groupby('molecule_name'))
        ])
        self.alternate_direction = alternate_direction

    def _add_targets(self, index, data):
        mol_name = self.metadata.iloc[index]['molecule_name']
        rows = self.source_data[mol_name].copy()
        

        if self.alternate_direction:
            row_tmp = rows.copy()
            rand_ind = np.random.choice(rows.index, size=len(rows)//2, replace=False)
            rows.loc[rand_ind, 'atom_index_1'] = row_tmp.loc[rand_ind, 'atom_index_0']
            rows.loc[rand_ind, 'atom_index_0'] = row_tmp.loc[rand_ind, 'atom_index_1']
        else:
            # make both direction:
            rows = rows.append(
                rows.rename({'atom_index_1': 'atom_index_0', 
                            'atom_index_0': 'atom_index_1'}, 
                            axis=1),
                sort=False
            )
        rows = rows.sort_values(['atom_index_0',
                               'atom_index_1'])
        
        data.couples_ind = torch.cat([
            torch.tensor(rows[['atom_index_0',
                               'atom_index_1']].values,
                         dtype=torch.long)
        ])            
        
        try:
            data.y = torch.tensor(
                rows['scalar_coupling_constant'].values.reshape(-1, 1),
                dtype=torch.float)
        except:
            data.y = torch.zeros((len(rows), 1), dtype=torch.float)
        
        data.type = torch.tensor(
            rows['type'].map(constants.TYPES_DICT).values,
            dtype=torch.long)
        
        data.sample_weight = torch.tensor(
            rows['type'].map(constants.TYPES_WEIGHTS).values,
            dtype=torch.float)
        
        data.mol_ind = torch.tensor([index], dtype=torch.long)
        return data




def mol_to_data(mol: openbabel.OBMol):
    """Extract data from OB Mol"""
    x = []
    pos = []
    for atom in openbabel.OBMolAtomIter(mol):
        x.append([
            *[atom.GetAtomicNum() == i
              for i in [1, 6, 7, 8, 9]],  # One hot atom type:H,C,N,O,F
            # atom.GetAtomicNum(), redundant
            atom.IsHbondAcceptor(),
            atom.IsHbondDonor(),
            atom.IsAromatic(),
            *[atom.GetHyb() == i for i in range(7)],  # One hot hybridization
            *[atom.ExplicitHydrogenCount() == i for i in range(4)],  # One hot hybridization
        ])
        pos.append([atom.GetX(), atom.GetY(), atom.GetZ()])
    edge_index = []
    edge_attr = []
    for bond in openbabel.OBMolBondIter(mol):
        atom_a = bond.GetBeginAtomIdx() - 1
        atom_b = bond.GetEndAtomIdx() - 1
        edge_index.extend([
            [atom_a, atom_b],
            [atom_b, atom_a],  # both direction
        ])
        edge_attr.extend(
            [[
                # bond.GetLength(),
                bond.IsSingle(),
                bond.IsDouble(),
                bond.IsTriple(),
                bond.IsAromatic()
            ]] * 2
        )
    edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = None
    res = FP16_Data(x=torch.tensor(x, dtype=torch.float),
               edge_index=edge_index,
               edge_attr=edge_attr,
               y=y,
               pos=torch.tensor(pos, dtype=torch.float))
    return res


def mol_to_data_v2(mol: openbabel.OBMol):
    """Extract data from OB Mol"""
    x = []
    pos = []
    for atom in openbabel.OBMolAtomIter(mol):
        x.append([
            *[atom.GetAtomicNum() == i
              for i in [1, 6, 7, 8, 9]],  # One hot atom type:H,C,N,O,F
            # atom.GetAtomicNum(), redundant
            atom.IsHbondAcceptor(),
            atom.IsHbondDonor(),
            atom.IsAromatic(),
            *[atom.GetHyb() == i for i in range(7)],  # One hot hybridization
            *[atom.ExplicitHydrogenCount() == i for i in range(4)],  # One hot hybridization
            atom.IsChiral(),
            *[atom.MemberOfRingSize() == i for i in [0, *range(3,9)]],
            atom.IsAxial(),
        ])
        pos.append([atom.GetX(), atom.GetY(), atom.GetZ()])
    edge_index = []
    edge_attr = []
    for bond in openbabel.OBMolBondIter(mol):
        atom_a = bond.GetBeginAtomIdx() - 1
        atom_b = bond.GetEndAtomIdx() - 1
        edge_index.extend([
            [atom_a, atom_b],
            [atom_b, atom_a],  # both direction
        ])
        edge_attr.extend(
            [[
                bond.IsInRing(),
                bond.IsSingle(),
                bond.IsDouble(),
                bond.IsTriple(),
                bond.IsAromatic(),
                bond.IsInRing(),
                bond.IsRotor(),
                bond.IsUp(),
                bond.IsDown(),
                bond.IsWedge(),
                bond.IsHash(),
            ]] * 2
        )
    edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = None
    res = FP16_Data(x=torch.tensor(x, dtype=torch.float),
               edge_index=edge_index,
               edge_attr=edge_attr,
               y=y,
               pos=torch.tensor(pos, dtype=torch.float))
    return res