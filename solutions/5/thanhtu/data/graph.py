"""Abstract classes and utility operations for building graph representations and
data loaders (known as Sequence objects in Keras).

Most users will not need to interact with this module."""
from abc import ABCMeta, abstractmethod
from operator import itemgetter
import numpy as np
from .utils import expand_1st, to_list
from monty.json import MSONable
from . import local_env
from inspect import signature
from pymatgen.analysis.local_env import NearNeighbors

class StructureGraph(MSONable):
    """
    This is a base class for converting converting structure into graphs or model inputs
    Methods to be implemented are follows:
        1. convert(self, structure)
            This is to convert a structure into a graph dictionary
        2. get_input(self, structure)
            This method convert a structure directly to a model input
        3. get_flat_data(self, graphs, targets)
            This method process graphs and targets pairs and output model input list.

    """

    # TODO (wardlt): Consider making "num_*_features" funcs to simplify making a MEGNet model

    def __init__(self,
                 nn_strategy=None,
                 atom_converter=None,
                 bond_converter=None,
                 **kwargs):

        if isinstance(nn_strategy, str):
            strategy = local_env.get(nn_strategy)
            parameters = signature(strategy).parameters
            param_dict = {i: j.default for i, j in parameters.items()}
            for i, j in kwargs.items():
                if i in param_dict:
                    setattr(self, i, j)
                    param_dict.update({i: j})
            self.nn_strategy = strategy(**param_dict)
        elif isinstance(nn_strategy, NearNeighbors):
            self.nn_strategy = nn_strategy
        elif nn_strategy is None:
            self.nn_strategy = None
        else:
            raise RuntimeError("Strategy not valid")

        self.atom_converter = atom_converter
        self.bond_converter = bond_converter
        if self.atom_converter is None:
            self.atom_converter = self._get_dummy_converter()
        if self.bond_converter is None:
            self.bond_converter = self._get_dummy_converter()

    def convert(self, structure, state_attributes=None):
        """
        Take a pymatgen structure and convert it to a index-type graph representation
        The graph will have node, distance, index1, index2, where node is a vector of Z number
        of atoms in the structure, index1 and index2 mark the atom indices forming the bond and separated by
        distance.
        For state attributes, you can set structure.state = [[xx, xx]] beforehand or the algorithm would
        take default [[0, 0]]

        Args:
            state_attributes: (list) state attributes
            structure: (pymatgen structure)
            (dictionary)
        """
        state_attributes = state_attributes or [[0, 0]]
        index1 = []
        index2 = []
        bonds = []
        if self.nn_strategy is None:
            raise RuntimeError("NearNeighbor strategy is not provided!")
        for n, neighbors in enumerate(self.nn_strategy.get_all_nn_info(structure)):
            index1.extend([n] * len(neighbors))
            for neighbor in neighbors:
                index2.append(neighbor['site_index'])
                bonds.append(neighbor['weight'])

        atoms = [i.specie.Z for i in structure]

        if np.size(np.unique(index1)) < len(atoms):
            raise RuntimeError("Isolated atoms found in the structure")
        else:
            return {'atom': np.array(atoms, dtype='int32').tolist(),
                    'bond': bonds,
                    'state': state_attributes,
                    'index1': index1,
                    'index2': index2
                    }

    def __call__(self, structure):
        return self.convert(structure)

    def get_input(self, structure):
        """
        Turns a structure into model input
        """
        graph = self.convert(structure)
        return self.graph_to_input(graph)

    def graph_to_input(self, graph):
        """
        Turns a graph into model input

        Args:
            (dict): Dictionary description of the graph
        Return:
            ([np.ndarray]): Inputs in the form needed by MEGNet
        """
        gnode = [0] * len(graph['atom'])
        gbond = [0] * len(graph['index1'])

        return [expand_1st(self.atom_converter.convert(graph['atom'])),
                expand_1st(self.bond_converter.convert(graph['bond'])),
                expand_1st(np.array(graph['state'])),
                expand_1st(np.array(graph['index1'])),
                expand_1st(np.array(graph['index2'])),
                expand_1st(np.array(gnode)),
                expand_1st(np.array(gbond))]

    def get_flat_data(self, graphs, targets=None):
        """
        Expand the graph dictionary to form a list of features and targets tensors.
        This is useful when the model is trained on assembled graphs on the fly.

        Args:
            graphs: (list of dictionary) list of graph dictionary for each structure
            targets: (list of float or list) Optional: corresponding target
                values for each structure

        Returns:
            tuple(node_features, edges_features, global_values, index1, index2, targets)
        """

        output = []  # Will be a list of arrays

        # Convert the graphs to matrices
        for feature in ['atom', 'bond', 'state', 'index1', 'index2']:
            output.append([np.array(x[feature]) for x in graphs])

        # If needed, add the targets
        if targets is not None:
            output.append([to_list(t) for t in targets])

        return tuple(output)

    @staticmethod
    def _get_dummy_converter():
        return DummyConverter()

    def as_dict(self):
        all_dict = super().as_dict()
        if 'nn_strategy' in all_dict:
            nn_strategy = all_dict.pop('nn_strategy')
            all_dict.update({'nn_strategy': local_env.serialize(nn_strategy)})
        return all_dict

    @classmethod
    def from_dict(cls, d):
        if 'nn_strategy' in d:
            nn_strategy = d.pop('nn_strategy')
            nn_strategy_obj = local_env.deserialize(nn_strategy)
            d.update({'nn_strategy': nn_strategy_obj})
            return super().from_dict(d)
        return super().from_dict(d)


class DistanceConverter(MSONable):
    """
    Base class for distance conversion. The class needs to have a convert method.
    """
    def convert(self, d):
        raise NotImplementedError


class DummyConverter(DistanceConverter):
    """
    Dummy converter as a placeholder
    """
    def convert(self, d):
        return d


class GaussianDistance(DistanceConverter):
    """
    Expand distance with Gaussian basis sit at centers and with width 0.5.

    Args:
        centers: (np.array)
        width: (float)
    """

    def __init__(self, centers=np.linspace(0, 5, 100), width=0.5):
        self.centers = centers
        self.width = width

    def convert(self, d):
        """
        expand distance vector d with given parameters

        Args:
            d: (1d array) distance array

        Returns
            (matrix) N*M matrix with N the length of d and M the length of centers
        """
        d = np.array(d)
        return np.exp(-(d[:, None] - self.centers[None, :]) ** 2 / self.width ** 2)


class MoorseLongRange(DistanceConverter):
    """
    This is an attempt to implement a Moorse/long range interactomic potential like
    distance expansion. The distance is expanded with this basis at different equilibrium
    bond distance, r_eq. It is still a work in progress. Do not use if you do not know
    much about the parameters
    ref: https://en.wikipedia.org/wiki/Morse/Long-range_potential#Function

    Args:
        d_e: (float) dissociate energy
        r_ref: (float) reference bond length
        r_eq: (list) equilibrium bond length
        p: (int) exponential term in the original equation, see ref
        q: (int) exponential term in the original equaiton, see ref
        cm: (list) long range coefficients in u_LR = \Sigma_i_N (cm_i / r^i)
        betas: (list) parameters determining the transition between long range and short range
    """
    def __init__(self, d_e=1, r_ref=2, r_eq=[1, 2, 3],
                 p=2, q=2, cm=[1, 2, 3, 4],
                 betas=[0.1, 0.2, 0.3, 0.4]):
        self.d_e = d_e
        self.r_ref = r_ref
        self.r_eq = np.array(r_eq)
        self.p = p
        self.q = q
        self.cm = np.array(cm).ravel()
        self.n_cm = len(self.cm)
        self.betas = np.array(betas).ravel()

    def convert(self, d):
        return self.d_e * (1 - self.u(d)[:, None] / self.u(self.r_eq)[None, :] *
                           np.exp(-self.beta(d) * self.y(d[:, None], self.r_eq[None, :], self.p))) ** 2

    def u(self, r):
        m_i = np.arange(1, self.n_cm + 1)
        if np.size(r) == 1:
            return np.sum(self.cm / r**m_i)
        return np.sum(self.cm[None, :] / r[:, None]**m_i[None, :], axis=1).ravel()

    @staticmethod
    def y(r, r_ref, p):
        return (r**p - r_ref**p) / (r**p + r_ref**p)

    def beta(self, r):
        y_p_ref = self.y(r, self.r_ref, self.p)
        y_q_ref = self.y(r, self.r_ref, self.q)
        return self.beta_inf[None, :] * y_p_ref[:, None] + (1 - y_p_ref[:, None]) * \
            np.sum(self.betas[None, :] * y_q_ref[:,  None] ** np.arange(0, len(self.betas))[None, :], axis=1).ravel()[:, None]

    @property
    def beta_inf(self):
        return np.log(2*self.d_e/self.u(self.r_eq))


