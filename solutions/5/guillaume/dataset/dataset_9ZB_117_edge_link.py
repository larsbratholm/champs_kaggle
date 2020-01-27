from numpy import load

from common import *
import itertools
import scipy.linalg
from torch_geometric.data import Data
import torch
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import os
import torch
from torch.utils.data import Dataset
import numba
from dscribe.descriptors import ACSF, SOAP, MBTR, LMBTR
from dscribe.core.system import System
from scipy.spatial.transform import Rotation as R
import h5py

def apply_random_rotation(vectors):
    r = R.from_euler('zxy', np.random.random(size = (1, 3)) * 360, degrees=True)
    return r.apply(vectors)

SYMBOL=['H', 'C', 'N', 'O', 'F']

ACSF_GENERATOR = ACSF(
    species=SYMBOL,
    rcut=6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)

rcut = 6.0
nmax = 8
lmax = 6

SOAP_GENERATOR = SOAP(
    species=SYMBOL,
    periodic=False,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
)

# Setup
MBTR_GENERATOR = MBTR(
    species=SYMBOL,
    k1={
        "geometry": {"function": "atomic_number"},
        "grid": {"min": 0, "max": 8, "n": 100, "sigma": 0.1},
    },
    k2={
        "geometry": {"function": "inverse_distance"},
        "grid": {"min": 0, "max": 1, "n": 100, "sigma": 0.1},
        "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
    },
    k3={
        "geometry": {"function": "cosine"},
        "grid": {"min": -1, "max": 1, "n": 100, "sigma": 0.1},
        "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
    },
    periodic=False,
    normalization="l2_each",
)

LMBTR_GENERATOR = LMBTR(
    species=SYMBOL,
    k2={
        "geometry": {"function": "distance"},
        "grid": {"min": 0, "max": 5, "n": 100, "sigma": 0.1},
        "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
    },
    k3={
        "geometry": {"function": "angle"},
        "grid": {"min": 0, "max": 180, "n": 100, "sigma": 0.1},
        "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
    },
    periodic=False,
    normalization="l2_each",
)

@numba.njit
def build_bond_vector(connectivity, xyz):
    bond_vectors = np.zeros((connectivity.shape[0], 3))
    
    for bond_i in range(connectivity.shape[0]):
        atom_i, atom_j = connectivity[bond_i, 0], connectivity[bond_i, 1]
        bond_vectors[bond_i] = xyz[atom_j] - xyz[atom_j]
        bond_i += 1

    return bond_vectors

script_dir = os.path.abspath(os.path.dirname(__file__))
permute_results = pd.read_pickle(script_dir + '/../processed_data/permute_results.9Y.pkl')
features = permute_results.loc[permute_results['loss'] > -2.09]
features = features.loc[~features['feature_name'].str.contains('dist_H')]

names = pd.read_pickle(script_dir + '/../processed_data/names_117.pkl')

class EdgeBasedDataset(Dataset):
    def __init__(self, name = 'train'):
        super().__init__()
        self.name = name
        
        self.global_features = h5py.File(script_dir + '/../processed_data/global_116.h5', mode = 'r')
        self.atom_features = h5py.File(script_dir + '/../processed_data/atom_116.h5', mode = 'r')
        self.bond_features = h5py.File(script_dir + '/../processed_data/bond_116.h5', mode = 'r')
        
        self.global_embedding_count = self.global_features['embeddings'][:, -1].max() + 1
        self.atom_embedding_count = self.atom_features['embeddings'][:, -1].max() + 1
        self.bond_ebedding_count = self.bond_features['embeddings'][:, -1].max() + 1
        
        self.atom_descriptors = pd.read_pickle(script_dir + '/../processed_data/atom_descriptors_117.pkl')
        self.bond_descriptors = pd.read_pickle(script_dir + '/../processed_data/bond_descriptors_117.pkl')
        
        self.dataset = pd.read_pickle(script_dir + '/../processed_data/dataset_descriptor_117.pkl')
        
        self.cycles = pd.read_pickle(script_dir + '/../processed_data/cycles_117.pkl').set_index('molecule_id')
        self.edges_connectivity = pd.read_hdf(script_dir + '/../processed_data/edges_connectivity_117.pkl', 'data').set_index('molecule_id')
 
        if self.name == 'train':
            self.length = self.dataset.loc[self.dataset['dataset'] == 'train'].shape[0]
            self.molecules = list(self.dataset.loc[self.dataset['dataset'] == 'train', 'molecule_name'])
            self.molecules_ids = list(self.dataset.loc[self.dataset['dataset'] == 'train', 'molecule_id'])

        elif self.name == 'test':
            self.length = self.dataset.loc[self.dataset['dataset'] == 'test'].shape[0]
            self.molecules = list(self.dataset.loc[self.dataset['dataset'] == 'test', 'molecule_name'])
            self.molecules_ids = list(self.dataset.loc[self.dataset['dataset'] == 'test', 'molecule_id'])
        
        self.global_features.close()
        self.atom_features.close()
        self.bond_features.close()
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.global_features = h5py.File(script_dir + '/../processed_data/global_116.h5', mode = 'r')
        self.atom_features = h5py.File(script_dir + '/../processed_data/atom_116.h5', mode = 'r')
        self.bond_features = h5py.File(script_dir + '/../processed_data/bond_116.h5', mode = 'r')
        
        molecule_id = self.molecules_ids[idx]
        molecule = self.molecules[idx]
        
        atom_descriptor = self.atom_descriptors.loc[molecule_id]
        bond_descriptor = self.bond_descriptors.loc[molecule_id]
        #bond_descriptor = bond_descriptor.loc[bond_descriptor['bond_distance'] <= 3]
        
        # Cycles
        if molecule_id in self.cycles.index:
            cycles = self.cycles.loc[molecule_id]
        else:
            cycles = pd.DataFrame(columns = self.cycles.columns)
            
        cycles_edge_index = cycles['edge_index'].values.astype(np.int64)
        cycles_id = cycles['cycle_id'].values.astype(np.int64)
        
        # Edge connectivity
        edges_connectivity = self.edges_connectivity.loc[molecule_id]
        edges_connectivity_ids = np.copy(edges_connectivity[['edge_index_0', 'edge_index_1']].values.astype(np.int64).T)
        edges_connectivity_vectors_0 = edges_connectivity[['vx_0', 'vy_0', 'vz_0']].values
        edges_connectivity_vectors_1 = edges_connectivity[['vx_1', 'vy_1', 'vz_1']].values
        
        edges_connectivity_feature_1 = np.sqrt(np.square(edges_connectivity_vectors_0).sum(axis = 1)).reshape(-1, 1)
        edges_connectivity_feature_2 = np.sqrt(np.square(edges_connectivity_vectors_1).sum(axis = 1)).reshape(-1, 1)
        edges_connectivity_feature_3 = np.sqrt(np.square(edges_connectivity_vectors_0 + edges_connectivity_vectors_1).sum(axis = 1)).reshape(-1, 1)
        edges_connectivity_feature_0 = (edges_connectivity_vectors_0 * edges_connectivity_vectors_1).sum(axis = 1).reshape(-1, 1) / edges_connectivity_feature_1 / edges_connectivity_feature_2
        edges_connectivity_features = np.concatenate([edges_connectivity_feature_0, edges_connectivity_feature_1, edges_connectivity_feature_2, edges_connectivity_feature_3], axis = 1)
        
        

        atom = list(atom_descriptor['atom'])
        xyz = atom_descriptor[['x', 'y', 'z']].values
        xyz = apply_random_rotation(xyz)

        connectivity = bond_descriptor[['atom_index_0', 'atom_index_1']].values

        global_feature_numeric = np.copy(self.global_features['numeric'][molecule_id].reshape(1, -1))
        global_feature_embeddings = np.copy(self.global_features['embeddings'][molecule_id].reshape(1, -1))
        
        atom_indexes = atom_descriptor['index'].values
        atom_index_min = atom_indexes.min()
        atom_index_max = atom_indexes.max()
        
        atom_feature_numeric = np.copy(self.atom_features['numeric'][atom_index_min : atom_index_max + 1][atom_indexes - atom_index_min])
        atom_feature_embeddings = np.copy(self.atom_features['embeddings'][atom_index_min : atom_index_max + 1][atom_indexes - atom_index_min])
        
        bond_indexes = bond_descriptor['index'].values
        bond_index_min = bond_indexes.min()
        bond_index_max = bond_indexes.max()
        
        bond_feature_numeric = np.copy(self.bond_features['numeric'][bond_index_min : bond_index_max + 1][bond_indexes - bond_index_min])
        bond_feature_embeddings = np.copy(self.bond_features['embeddings'][bond_index_min : bond_index_max + 1][bond_indexes - bond_index_min])
        
        self.global_features.close()
        self.atom_features.close()
        self.bond_features.close()
        
        # chemical descriptors
        atom = System(symbols = atom, positions=xyz)
        acsf = ACSF_GENERATOR.create(atom)
        
        atom_feature_numeric = np.concatenate([atom_feature_numeric, xyz, acsf], axis = 1)
        
        bond_vectors = build_bond_vector(connectivity, xyz)
        
        bond_feature_numeric = np.concatenate([bond_feature_numeric, bond_vectors], axis = 1)
        
        if self.name == 'train':
            # Target
            target = bond_descriptor['scalar_coupling_constant'].values.reshape(-1, 1)
            target_mask = (bond_descriptor['type'] != 'VOID').values.reshape(-1, 1)
            target_types = bond_descriptor['type_id'].values.reshape(-1, 1)
            target_idx = bond_descriptor['edge_index'].values.reshape(-1, 1)


            # data
            data = Data(
                x_numeric = torch.tensor(atom_feature_numeric, dtype = torch.float32),
                x_embeddings = torch.tensor(atom_feature_embeddings, dtype = torch.int64),
                
                edge_attr_numeric = torch.tensor(bond_feature_numeric, dtype = torch.float32),
                edge_attr_embeddings = torch.tensor(bond_feature_embeddings, dtype = torch.int64),
                
                u_numeric = torch.tensor(global_feature_numeric, dtype = torch.float32),
                u_embeddings = torch.tensor(global_feature_embeddings, dtype = torch.int64),
                
                edge_index = torch.tensor(connectivity.T),
                
                num_nodes = atom_feature_numeric.shape[0],
                
                molecule_ids = torch.tensor([molecule_id], dtype = torch.int64),
                
                y = torch.tensor(target, dtype = torch.float32),
                y_mask = torch.tensor(target_mask, dtype = torch.float32),
                y_types = torch.tensor(target_types, dtype = torch.int64),
                y_idx = torch.tensor(target_idx, dtype = torch.int32),
                
                cycles_edge_index = torch.tensor(cycles_edge_index),
                cycles_id = torch.tensor(cycles_id),
                
                edges_connectivity_ids = torch.tensor(edges_connectivity_ids),
                edges_connectivity_features = torch.tensor(edges_connectivity_features, dtype = torch.float32),
            )
            
            
            inputs = [
                data.u_embeddings, data.x_embeddings, data.edge_attr_embeddings,
                data.u_numeric, data.x_numeric, data.edge_attr_numeric,
            ]
                      
            for input_i in range(6):
                index_names = list(features.loc[(features['description_i'] == input_i), 'feature_name'].values)
                index = np.array([names[input_i].index(name) for name in index_names])
                
                inputs[input_i] = inputs[input_i][:, index]
                
                if inputs[input_i].shape[1] == 0:
                    inputs[input_i] = torch.zeros((inputs[input_i].shape[0], 1), dtype = inputs[input_i].dtype)

            data.u_embeddings = inputs[0]
            data.x_embeddings = inputs[1]
            data.edge_attr_embeddings = inputs[2]
            data.u_numeric = inputs[3]
            data.x_numeric = inputs[4]
            data.edge_attr_numeric = inputs[5]
            
        else:
            # Target
            target_mask = (bond_descriptor['type'] != 'VOID').values.reshape(-1, 1)
            target_types = bond_descriptor['type_id'].values.reshape(-1, 1)
            target_idx = bond_descriptor['edge_index'].values.reshape(-1, 1)
            
            # data
            data = Data(
                x_numeric = torch.tensor(atom_feature_numeric, dtype = torch.float32),
                x_embeddings = torch.tensor(atom_feature_embeddings, dtype = torch.int64),
                
                edge_attr_numeric = torch.tensor(bond_feature_numeric, dtype = torch.float32),
                edge_attr_embeddings = torch.tensor(bond_feature_embeddings, dtype = torch.int64),
                
                u_numeric = torch.tensor(global_feature_numeric, dtype = torch.float32),
                u_embeddings = torch.tensor(global_feature_embeddings, dtype = torch.int64),
                
                edge_index = torch.tensor(connectivity.T),
                
                num_nodes = atom_feature_numeric.shape[0],
                
                molecule_ids = torch.tensor([molecule_id], dtype = torch.int64),
                
                y_mask = torch.tensor(target_mask, dtype = torch.float32),
                y_types = torch.tensor(target_types, dtype = torch.int64),
                y_idx = torch.tensor(target_idx, dtype = torch.int32),
                
                cycles_edge_index = torch.tensor(cycles_edge_index),
                cycles_id = torch.tensor(cycles_id),
                
                edges_connectivity_ids = torch.tensor(edges_connectivity_ids),
                edges_connectivity_features = torch.tensor(edges_connectivity_features, dtype = torch.float32),
            )
            
            inputs = [
                data.u_embeddings, data.x_embeddings, data.edge_attr_embeddings,
                data.u_numeric, data.x_numeric, data.edge_attr_numeric,
            ]
            for input_i in range(6):
                index_names = list(features.loc[(features['description_i'] == input_i), 'feature_name'].values)
                index = np.array([names[input_i].index(name) for name in index_names])

                inputs[input_i] = inputs[input_i][:, index]

                if inputs[input_i].shape[1] == 0:
                    inputs[input_i] = torch.zeros((inputs[input_i].shape[0], 1), dtype = inputs[input_i].dtype)

            data.u_embeddings = inputs[0]
            data.x_embeddings = inputs[1]
            data.edge_attr_embeddings = inputs[2]
            data.u_numeric = inputs[3]
            data.x_numeric = inputs[4]
            data.edge_attr_numeric = inputs[5]

        return data
    

from torch_geometric.data import Batch
import torch_geometric

def from_data_list(data_list, follow_batch=[]):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects.
    The assignment vector :obj:`batch` is created on the fly.
    Additionally, creates assignment batch vectors for each key in
    :obj:`follow_batch`."""

    keys = [set(data.keys) for data in data_list]
    keys = list(set.union(*keys))
    assert 'batch' not in keys

    batch = Batch()

    for key in keys:
        batch[key] = []

    for key in follow_batch:
        batch['{}_batch'.format(key)] = []

    cumsum = {}
    batch.batch = []
    for i, data in enumerate(data_list):
        for key in data.keys:
            item = data[key] + cumsum.get(key, 0)
            if key in cumsum:
                if key == 'edge_index':
                    cumsum[key] += data['x_numeric'].shape[0]
                
                if key == 'subgraphs_nodes':
                    cumsum[key] += data['x_numeric'].shape[0]
                if key == 'subgraphs_edges':
                    cumsum[key] += data['edge_attr_numeric'].shape[0]
                if key == 'subgraphs_connectivity':
                    cumsum[key] += data['subgraphs_nodes'].shape[0]
                    
                if key == 'subgraphs_nodes_path_batch':
                    cumsum[key] += data['subgraphs_nodes_path_batch'].max() + 1
                if key == 'subgraphs_edges_path_batch':
                    cumsum[key] += data['subgraphs_edges_path_batch'].max() + 1
                if key == 'subgraphs_global_path_batch':
                    cumsum[key] += 1
                    
                if key == 'cycles_id':
                    if data['cycles_id'].shape[0] > 0:
                        cumsum[key] += data['cycles_id'].max() + 1
                        
                if key == 'cycles_edge_index':
                    cumsum[key] += data['edge_attr_numeric'].shape[0]
                    
                if key == 'edges_connectivity_ids':
                    cumsum[key] += data['edge_attr_numeric'].shape[0]
                    
            else:
                if key == 'edge_index':
                    cumsum[key] = data['x_numeric'].shape[0]
                
                if key == 'subgraphs_nodes':
                    cumsum[key] = data['x_numeric'].shape[0]
                if key == 'subgraphs_edges':
                    cumsum[key] = data['edge_attr_numeric'].shape[0]
                if key == 'subgraphs_connectivity':
                    cumsum[key] = data['subgraphs_nodes'].shape[0]
                
                
                if key == 'subgraphs_nodes_path_batch':
                    cumsum[key] = data['subgraphs_nodes_path_batch'].max() + 1
                if key == 'subgraphs_edges_path_batch':
                    cumsum[key] = data['subgraphs_edges_path_batch'].max() + 1
                if key == 'subgraphs_global_path_batch':
                    cumsum[key] = 1
                    
                    
                if key == 'cycles_id':
                    if data['cycles_id'].shape[0] > 0:
                        cumsum[key] = data['cycles_id'].max() + 1
                        
                if key == 'cycles_edge_index':
                    cumsum[key] = data['edge_attr_numeric'].shape[0]
                    
                if key == 'edges_connectivity_ids':
                    cumsum[key] = data['edge_attr_numeric'].shape[0]


            batch[key].append(item)
            
        for key in follow_batch:
            size = data[key].size(data.__cat_dim__(key, data[key]))
            item = torch.full((size, ), i, dtype=torch.long)
            batch['{}_batch'.format(key)].append(item)

        num_nodes = data.num_nodes
        if num_nodes is not None:
            item = torch.full((num_nodes, ), i, dtype=torch.long)
            batch.batch.append(item)

    if num_nodes is None:
        batch.batch = None

    for key in batch.keys:
        item = batch[key][0]
        if torch.is_tensor(item):
            if key not in ['subgraphs_connectivity', 'edges_connectivity_ids']:
                batch[key] = torch.cat(
                    batch[key], dim=data_list[0].__cat_dim__(key, item))
            else:
                batch[key] = torch.cat(
                    batch[key], dim=-1)
        elif isinstance(item, int) or isinstance(item, float):
            batch[key] = torch.tensor(batch[key])
        else:
            raise ValueError('Unsupported attribute type.')

    # Copy custom data functions to batch (does not work yet):
    # if data_list.__class__ != Data:
    #     org_funcs = set(Data.__dict__.keys())
    #     funcs = set(data_list[0].__class__.__dict__.keys())
    #     batch.__custom_funcs__ = funcs.difference(org_funcs)
    #     for func in funcs.difference(org_funcs):
    #         setattr(batch, func, getattr(data_list[0], func))

    if torch_geometric.is_debug_enabled():
        batch.debug()

    return batch.contiguous()

from itertools import chain
class DataParallel(torch.nn.DataParallel):
    r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given :attr:`module` by
    splitting a list of :class:`torch_geometric.data.Data` objects and copying
    them as :class:`torch_geometric.data.Batch` objects to each device.
    In the forward pass, the module is replicated on each device, and each
    replica handles a portion of the input.
    During the backwards pass, gradients from each replica are summed into the
    original module.

    The batch size should be larger than the number of GPUs used.

    The parallelized :attr:`module` must have its parameters and buffers on
    :obj:`device_ids[0]`.

    .. note::

        You need to use the :class:`torch_geometric.data.DataListLoader` for
        this module.

    Args:
        module (Module): Module to be parallelized.
        device_ids (list of int or torch.device): CUDA devices.
            (default: all devices)
        output_device (int or torch.device): Device location of output.
            (default: :obj:`device_ids[0]`)
    """

    def __init__(self, module, device_ids=None, output_device=None, follow_batch = []):
        super(DataParallel, self).__init__(module, device_ids, output_device)
        self.follow_batch = follow_batch
        self.src_device = torch.device("cuda:{}".format(self.device_ids[0]))

    def forward(self, data_list):
        """"""
        if len(data_list) == 0:
            warnings.warn('DataParallel received an empty data list, which '
                          'may result in unexpected behaviour.')
            return None

        if not self.device_ids or len(self.device_ids) == 1:  # Fallback
            data = from_data_list(data_list, follow_batch = self.follow_batch).to(self.src_device)
            return self.module(data)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device:
                raise RuntimeError(
                    ('Module must have its parameters and buffers on device '
                     '{} but found one of them on device {}.').format(
                         self.src_device, t.device))

        inputs = self.scatter(data_list, self.device_ids)
        print([e.x_numeric.device for e in inputs])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        print([[f.device.index for f in e.parameters()] for e in replicas])
        outputs = self.parallel_apply(replicas, inputs, None)
        return self.gather(outputs, self.output_device)

    def scatter(self, data_list, device_ids):
        num_devices = min(len(device_ids), len(data_list))

        count = torch.tensor([data.num_nodes for data in data_list])
        cumsum = count.cumsum(0)
        cumsum = torch.cat([cumsum.new_zeros(1), cumsum], dim=0)
        device_id = num_devices * cumsum.to(torch.float) / cumsum[-1].item()
        device_id = (device_id[:-1] + device_id[1:]) / 2.0
        device_id = device_id.to(torch.long)  # round.
        split = device_id.bincount().cumsum(0)
        split = torch.cat([split.new_zeros(1), split], dim=0)
        split = torch.unique(split, sorted=True)
        split = split.tolist()

        return [
            from_data_list(data_list[split[i]:split[i + 1]], follow_batch = self.follow_batch).to(
                torch.device('cuda:{}'.format(device_ids[i])))
            for i in range(len(split) - 1)
        ]

class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 follow_batch=[],
                 **kwargs):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: from_data_list(
                data_list, follow_batch),
            **kwargs)
        
class Dispatcher(torch.nn.Module):
    def __init__(self, model):
        super(Dispatcher, self).__init__()
        self.model = model
        
    def forward(self, batch):
        out = model(
            #[batch.x, batch.x_acsf],
            [batch.x_numeric],
            batch.x_embeddings,
            batch.edge_index, 
            [batch.edge_attr_numeric], 
            batch.edge_attr_embeddings,
            #[batch.u, batch.u_mbtr], 
            [batch.u_numeric],
            batch.u_embeddings,
            batch.batch, 
            batch.y_types, 
            batch.edge_attr_numeric_batch, 
            batch.subgraphs_nodes, 
            batch.subgraphs_edges, 
            batch.subgraphs_connectivity, 
            batch.subgraphs_nodes_path_batch, 
            batch.subgraphs_edges_path_batch, 
            batch.subgraphs_global_path_batch,
            batch.subgraphs_nodes_batch, 
            batch.subgraphs_edges_batch
        )
        return out
#m = DataParallel(Dispatcher(model), follow_batch=['edge_attr_numeric', 'subgraphs_nodes', 'subgraphs_edges'])#.to('cuda')
#out = m(batch)