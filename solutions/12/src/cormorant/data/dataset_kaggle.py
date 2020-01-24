import numpy as np
import torch
import logging

from torch.utils.data import Dataset
from scipy.sparse import coo_matrix


class KaggleTrainDataset(Dataset):
    """
    Data structure for a "Predicting Molecular Properties" dataset.  Extends PyTorch Dataset.
    Does not download or porcess. Based upon a pre-processed dataset.

    Parameters
    ----------
    data : dict
        Dictionary of arrays containing molecular properties.
    num_pts : int, optional
        Desired number of points to include in the dataset.
        Default value, -1, uses all of the datapoints.
    shuffle : bool, optional
        If true, shuffle the points in the dataset.
    """
    # TODO: Rewrite with a better format and using torch.tensors() instead of np.arrays()
    def __init__(self, data, num_pts=-1, normalize=True, shuffle=True, subtract_thermo=True, additional_atom_features=None):

        self.data = data

        if num_pts < 0:
            self.num_pts = len(data['charges'])
        else:
            if num_pts > len(data['charges']):
                logging.warn('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, len(data['charges'])))
                self.num_pts = len(data['charges'])
            else:
                self.num_pts = num_pts

        # Fix by hand for dataset
        included_species = torch.tensor([1, 6, 7, 8, 9])
        self.included_species = included_species
        self.num_species = len(included_species)
        self.max_charge = max(included_species)
        if additional_atom_features is None:
            self.additional_atom_features = []
        else:
            self.additional_atom_features = list(additional_atom_features)

        self.coupling_subtypes = ["1JHC", "1JHN", "2JHH", "2JHC", "2JHN", "3JHH", "3JHC", "3JHN"]
        self.valid_labels = ['jj_%d_value' % i for i in range(1, 4)] + ['jj_all_value']
        self.valid_labels += ['%s_value' % label_i for label_i in self.coupling_subtypes]

        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}

        self._split_jj_couplings()

        # At the moment, the only statistics we need are for the JJ couplings,
        # so we won't automatically calculate the rest of the statistics.
        self.stats = {}
        self._calc_jj_stats()
        self.shuffle = shuffle
        self._build_perm()

    def _build_perm(self):
        """
        Builds a permutation of the indices, which is stored in self.perm

        Parameters
        ----------
        shuffle : bool
            If True, sets self.perm to a permutation matrix.  If not, to None.
        """
        if self.shuffle:
            self.perm = torch.randperm(len(self.data['charges']))[:self.num_pts]
        else:
            self.perm = None


    def _split_jj_couplings(self):
        jj_splits = {}
        jj_splits.update({'jj_'+str(key)+'_edge': [] for key in [1, 2, 3]})
        jj_splits.update({'jj_'+str(key)+'_value': [] for key in [1, 2, 3]})
        jj_splits.update({'jj_'+str(key)+'_label': [] for key in [1, 2, 3]})
        for jj_type, jj_edge, jj_value, jj_label in zip(self.data['jj_type'], self.data['jj_edge'], self.data['jj_value'], self.data['jj_label']):
            for jj_t in [1, 2, 3]:
                this_jj = (jj_type == jj_t)

                jj_splits['jj_'+str(jj_t)+'_edge'].append(jj_edge[this_jj, :])
                jj_splits['jj_'+str(jj_t)+'_value'].append(jj_value[this_jj])
                jj_splits['jj_'+str(jj_t)+'_label'].append(jj_label[this_jj])

        self.data.update(jj_splits)

        jj_subtype_splits = {}
        jj_subtype_splits.update({'%s_edge' % subtype_i: [] for subtype_i in self.coupling_subtypes})
        jj_subtype_splits.update({'%s_value' % subtype_i: [] for subtype_i in self.coupling_subtypes})
        for jj_label, jj_edge, jj_value in zip(self.data['jj_label'], self.data['jj_edge'], self.data['jj_value']):
            for i, subtype_i in enumerate(self.coupling_subtypes):
                this_subtype = (jj_label == subtype_i)
                jj_subtype_splits["%s_edge" % subtype_i].append(jj_edge[this_subtype, :])
                jj_subtype_splits["%s_value" % subtype_i].append(jj_value[this_subtype])
        self.data.update(jj_subtype_splits)

        for prefix in ["jj_1", "jj_2", "jj_3"]:
            for suffix in ['_edge', '_value', '_label']:
                key = prefix + suffix
                self.data[key] = np.array(self.data[key])

        for prefix in self.coupling_subtypes:
            for suffix in ['_edge', '_value']:
                key = prefix + suffix
                self.data[key] = np.array(self.data[key])
        
        jj_all_splits = {}
        jj_all_splits.update({'jj_all_edge': []})
        jj_all_splits.update({'jj_all_value': []})
        jj_all_splits.update({'jj_all_label': []})
        for jj_edge, jj_value, jj_label in zip(self.data['jj_edge'], self.data['jj_value'], self.data['jj_label']):
            jj_all_splits["jj_all_edge"].append(jj_edge)
            jj_all_splits["jj_all_value"].append(jj_value)
            jj_all_splits["jj_all_label"].append(jj_label)
        self.data.update(jj_all_splits)

        for suffix in ['_edge', '_value', '_label']:
            key = 'jj_all' + suffix
            self.data[key] = np.array(self.data[key])

    
    def _calc_jj_stats(self):
        for jj_target, jj_split in self.data.items():
            if jj_target not in self.valid_labels:
                continue

            jj_values_cat = np.concatenate(jj_split)
            self.stats[jj_target[:-6]] = (jj_values_cat.mean(), jj_values_cat.std())

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]

        temp_data = {key: val[idx] for key, val in self.data.items()}
        
        atom_features = ['charges', 'positions'] + self.additional_atom_features
        data = {key: torch.from_numpy(temp_data[key]) for key in atom_features}

        data['one_hot'] = torch.from_numpy(temp_data['charges']).unsqueeze(-1) == self.included_species.unsqueeze(0)

        num_atoms = len(temp_data['charges'])

        for jj_type in [1, 2, 3, 'all']:

            jj_str = 'jj_'+str(jj_type)
            rows, cols = temp_data[jj_str+'_edge'][:, 0], temp_data[jj_str+'_edge'][:, 1]
            values = temp_data[jj_str+'_value']

            jj_values = coo_matrix((values, (rows, cols)), shape=(num_atoms, num_atoms)).todense()
            data[jj_str] = torch.from_numpy(jj_values).unsqueeze(-1)

        for subtype_label in self.coupling_subtypes:
            rows, cols = temp_data[subtype_label+'_edge'][:, 0], temp_data[subtype_label+'_edge'][:, 1]
            values = temp_data[subtype_label+'_value']

            jj_values = coo_matrix((values, (rows, cols)), shape=(num_atoms, num_atoms)).todense()
            data[subtype_label] = torch.from_numpy(jj_values).unsqueeze(-1)

        return data
