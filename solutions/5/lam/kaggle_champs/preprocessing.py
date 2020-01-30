import numpy as np
import torch
from torch_geometric.utils import remove_self_loops
from scipy.stats import ortho_group
import torch_geometric.transforms as T


class AddVirtualEdges(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        # Keep index to find bonds
        row, col = data.edge_index       
        data.bonds_edge_ind = row * (data.num_nodes-1) + col
        data.bonds_edge_ind[row < col] = data.bonds_edge_ind[row < col] - 1
        data.bonds_edge_ind = data.bonds_edge_ind.view(-1,1)
           
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index
        
        return data
    
    
class RandomRotation():
    def __call__(self, data):
        proj_mat = torch.tensor(ortho_group.rvs(dim=3).T, dtype=torch.float)
        data.pos = torch.matmul(data.pos, proj_mat)
        if hasattr(data, 'aux_target'):
            assert data.aux_target.size()[1] == 10
            for i in range(3):
                data.aux_target[:, 3*i+1:3*i+4] = torch.matmul(data.aux_target[:, 3*i+1:3*i+4], proj_mat)
        return data
    
    
class AddEdgeDistanceAndDirection(T.Distance):
    def __init__(self, norm=False, max_value=None, cat=True, gaussian_base=False, unify_direction=False, dist_noise=0.):
        super(AddEdgeDistanceAndDirection, self).__init__(norm=norm,
                                                          max_value=max_value,
                                                          cat=cat)
        self.gaussian_base = gaussian_base
        self.unify_direction = unify_direction
        self.dist_noise = dist_noise

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
        if self.dist_noise > 0:
            noise = 1 + torch.randn_like(dist, dtype=dist.dtype) * self.dist_noise
            dist = dist * noise

        direction = (pos[col] - pos[row]) / dist
        if self.unify_direction:
            # unify direction to x > 0 
            direction[direction[:,0] < 0] = direction[direction[:,0] < 0] * (-1)

        if self.norm and dist.numel() > 0:
            dist = dist / dist.max() if self.max is None else self.max

        if self.gaussian_base:
            if self.norm:
                raise Exception('Cannot use both gaussian base and dist normalization')
            base = torch.linspace(0.2, 4, 20, dtype=torch.float).view(1, -1)
            dist = torch.exp(-(dist - base) ** 2 / 0.5 ** 2)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat(
                [pseudo,
                 dist.type_as(pseudo),
                 direction.type_as(pseudo)],
                dim=-1)
        else:
            data.edge_attr = torch.cat(
                [dist.type_as(pseudo),
                 direction.type_as(pseudo)], dim=-1)
        return data

    def __repr__(self):
        return '{}(norm={}, max_value={})'.format(self.__class__.__name__,
                                                  self.norm, self.max)


class SortTarget:
    def _get_index(self, data, row, col):
        idx = row * (data.num_nodes-1) + col
        idx[row < col] = idx[row < col] - 1
        return idx
    
    def __call__(self, data):
        target = torch.zeros((data.num_edges, data.y.size()[1]), dtype=torch.float)        
        weights = torch.zeros((data.num_edges), dtype=torch.float)        
        mask = torch.zeros((data.num_edges), dtype=torch.bool)      
        types = torch.zeros((data.num_edges), dtype=torch.long)
        
        row, col = data.couples_ind.transpose(1,0)
        indexes = self._get_index(data, row, col)
        
        mask[indexes] = True
        weights[indexes] = data.sample_weight
        target[indexes] = data.y
        types[indexes] = data.type
        
        data.mask = mask
        data.y = target[mask]
        data.sample_weight = weights[mask]
        data.type = types[mask]
        return data        