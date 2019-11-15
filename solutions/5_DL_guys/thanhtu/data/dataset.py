
import sys
sys.path.append('/home/ubuntu/datalab/kaggle/champ_scalar/reference/megnet')
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from build.data.utils import read_pickle_from_file
from dscribe.descriptors import ACSF
from dscribe.core.system import System
from build.data.data import MoleculaGraph, load_csv, make_graph
from time import time


DATA_DIR = '/home/ubuntu/datalab/kaggle/champ_scalar/data'


class MolecularGraphDataset(Dataset):

    def __init__(self, split, 
                       csv, 
                       mode,
                       augment=None):
        """Set Dataset for molecular graph
        
        Arguments:
            split {str} -- numpy split 
            csv {str} -- 'train' or 'test'
            mode {str} -- train
        
        Keyword Arguments:
            augment {[type]} -- [description] (default: {None})
        """

        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.augment = augment
        self.df = pd.read_csv(DATA_DIR + '/csv/%s.csv'%csv)
        if split is not None:
            self.id = np.load(DATA_DIR + '/split2/%s'%split,allow_pickle=True)
        else:
            self.id = self.df.molecule_name.unique()

    def __str__(self):
        string = ''\
        + '\tmode   = %s\n'%self.mode \
        + '\tsplit  = %s\n'%self.split \
        + '\tcsv    = %s\n'%self.csv \
        + '\tlen    = %d\n'%len(self)

        return string

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
        molecule_name = self.id[index]
        graph_file = f'{DATA_DIR}/structure/graph2/{molecule_name}.pickle'
        graph = read_pickle_from_file(graph_file)
        if 0:
            # 1JHC,     2JHC,     3JHC,     1JHN,     2JHN,     3JHN,     2JHH,     3JHH
            mask = np.zeros(len(graph['coupling'].type),np.bool)
            for t in ['2JHC' , '2JHN',  '2JHH']:
                mask += (graph['coupling'].type == COUPLING_TYPE.index(t))
        
            graph['coupling'].id = graph['coupling'].id[mask]
            graph['coupling'].contribution = graph['coupling'].contribution[mask]
            graph['coupling'].index = graph['coupling'].index[mask]
            graph['coupling'].type = graph['coupling'].type[mask]
            graph['coupling'].value = graph['coupling'].value[mask]
        return graph

def _collate_fn(batch):
    graphs = []
    targets = []
    batch_size = len(batch)
    offset = 0
    
    coupling_value = []
    coupling_atom_index  = []
    coupling_type_index  = []
    coupling_batch_index = []
    infor = []
    for b in range(batch_size):
        graph = batch[b]
        graphs.append(graph)
        
        num_coupling = len(graph['coupling'].value)
        coupling_value.append(graph['coupling'].value)
        coupling_atom_index.append(graph['coupling'].index+offset)
        coupling_type_index.append (graph['coupling'].type)
        coupling_batch_index.append(np.array([b]*num_coupling))
        infor.append(graph['coupling'].id)
        offset += len(graph['atom'])
    train_input = MoleculaGraph().get_flat_data(graphs)
    gnode = []
    for i, j in enumerate(train_input[0]):
        gnode += [i] * len(j)
    
    gbond = []
    for i, j in enumerate(train_input[1]):
        gbond += [i] * len(j)

    gnode = torch.from_numpy(np.ravel(gnode))
    gbond = torch.from_numpy(np.ravel(gbond))
    node = torch.from_numpy(np.concatenate(train_input[0])).float()
    edge = torch.from_numpy(np.concatenate(train_input[1])).float()
    state = torch.from_numpy(np.concatenate(train_input[2])).float()
    index1_temp = train_input[3]
    index2_temp = train_input[4]

    index1 = []
    index2 = []
    offset_ind = 0
    for ind1, ind2 in zip(index1_temp, index2_temp):
        index1 += [i + offset_ind for i in ind1]
        index2 += [i + offset_ind for i in ind2]
        offset_ind += (max(ind1) + 1)
    index1 = torch.from_numpy(np.ravel(index1)).long()
    index2 = torch.from_numpy(np.ravel(index2)).long()
    coupling_value = torch.from_numpy(np.concatenate(coupling_value)).float()

    targets = coupling_value

    coupling_index = np.concatenate([
        np.concatenate(coupling_atom_index),
        np.concatenate(coupling_type_index).reshape(-1,1),
        np.concatenate(coupling_batch_index).reshape(-1,1),
    ],-1)
    coupling_index = torch.from_numpy(coupling_index).long()
    inputs = [node, edge, state, index1, index2, gnode, gbond, coupling_index, infor]
    return inputs, targets


if __name__ == "__main__":
    dataset = MolecularGraphDataset(split='debug_split_by_mol.1000.npy',
                                    mode = 'train',
                                    csv = 'train',
                                    )
    train_dl = DataLoader(dataset, batch_size=16, 
                          shuffle=False, collate_fn=_collate_fn,
                          num_workers=0)
    # print(dataset[0])
    start = time()
    for inputs, targets in train_dl:
        print(time() - start)
        start = time()
        pass

    print('qsdf')


