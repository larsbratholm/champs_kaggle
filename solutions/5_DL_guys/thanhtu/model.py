from dataclasses import dataclass
from torch import nn
import torch
from data.dataset import *
from typing import List
from torch_scatter import scatter_mean, scatter_add, scatter_max
from layers import *
from constants import *


class MegnetLayer(nn.Module):

    def __init__(self, n_hidden_1=600, n_hidden_2=300):
        super(MegnetLayer, self).__init__()
    
        self.mlp_e = nn.Sequential(LinearBn(n_hidden_1 * 2, n_hidden_1, nn.Softplus()), 
                                   LinearBn(n_hidden_1, n_hidden_1, nn.Softplus()), 
                                   LinearBn(n_hidden_1, n_hidden_1/2, nn.Softplus()))

        self.mlp_v = nn.Sequential(LinearBn(n_hidden_2 *3, n_hidden_2 *2 , nn.Softplus()), 
                                   LinearBn(n_hidden_2 *2, n_hidden_2 *2, nn.Softplus()), 
                                   LinearBn(n_hidden_2 *2, n_hidden_2, nn.Softplus()))

        self.mlp_u = nn.Sequential(LinearBn(n_hidden_2 *3, n_hidden_2 *2 , nn.Softplus()), 
                                   LinearBn(n_hidden_2 *2, n_hidden_2 *2, nn.Softplus()), 
                                   LinearBn(n_hidden_2 *2, n_hidden_2, nn.Softplus()))
        self.edge_agg = EdgeAggregation(n_hidden_2, n_hidden_2*2)

    def phi_e(self, inputs):
        nodes, edges, states, index1, index2, gnode, gbond = inputs
        fs = torch.index_select(nodes, 0, index1)
        fr = torch.index_select(nodes, 0, index2)
        concate_node = torch.cat([fs, fr], dim = -1)
        u_expand = torch.index_select(states, 0, gbond)
        concated = torch.cat([concate_node, edges, u_expand], dim = -1)
        return edges + self.mlp_e(concated)
    
    def rho_e_v(self, e_k_p, inputs):
        nodes, edges, states, index1, index2, gnode, gbond = inputs
        return scatter_mean(e_k_p, index1, dim=0)

    def phi_v(self, v_e_i, inputs):
        nodes, edges, states, index1, index2, gnode, gbond = inputs
        u_expand = torch.index_select(states, 0,  gnode)
        edges = self.edge_agg(v_e_i, index1)
        concated = torch.cat([edges, nodes, u_expand], dim = -1)
        return nodes + self.mlp_v(concated)

    def rho_e_u(self, e_k_p, inputs):
        nodes, edges, states, index1, index2, gnode, gbond = inputs
        return scatter_mean(e_k_p, gbond, dim = 0)

    def phi_u(self, u_e, u_v, inputs):
        nodes, edges, states, index1, index2, gnode, gbond = inputs
        concated = torch.cat([u_e, u_v, states], dim = -1)
        return states + self.mlp_u(concated)
    
    def rho_v_u(self, v_i_p, inputs):
        nodes, edges, states, index1, index2, gnode, gbond = inputs
        return scatter_mean(v_i_p, gnode, dim = 0)

    def forward(self, inputs):
        e_k_p = self.phi_e(inputs)
        #v_e_i = self.rho_e_v(e_k_p, inputs)
        v_i_p = self.phi_v(e_k_p, inputs)
        u_e = self.rho_e_u(e_k_p, inputs)
        u_v = self.rho_v_u(v_i_p, inputs)
        u_p = self.phi_u(u_e, u_v, inputs)
        return v_i_p, e_k_p, u_p


class MegnetBloc(nn.Module):
    def __init__(self, n_hidden_1, n_hidden_2, has_ff = None):
        super(MegnetBloc, self).__init__()
        self.has_ff = has_ff
        self.ff_node = FeedForwardBloc([n_hidden_2, n_hidden_1, n_hidden_2])
        self.ff_edge = FeedForwardBloc([n_hidden_2, n_hidden_1, n_hidden_2])
        self.ff_state = FeedForwardBloc([n_hidden_2, n_hidden_1, n_hidden_2])

        self.megnet_layer = MegnetLayer(n_hidden_1, n_hidden_2)
                                        
    def forward(self, inputs):
        node, edge, state, index_1, index_2, gnode, gbond = inputs
        if self.has_ff:
            node = self.ff_node(node)
            edge = self.ff_edge(edge)
            state = self.ff_state(state)
        
        node, edge, state = self.megnet_layer([node, edge, state, index_1, index_2, gnode, gbond])

        return [node, edge, state, index_1, index_2, gnode, gbond]


class MegnetModel1(nn.Module):
    
    def __init__(self, 
                 nblocks = 7, 
                 n_hidden_1 =600, 
                 n_hidden_2 =300, 
                 n_hidden_3 =300, 
                 ntarget=8,
                 y_mean=torch.FloatTensor(COUPLING_TYPE_MEAN), 
                 y_std=torch.FloatTensor(COUPLING_TYPE_STD)):
                 
        super(MegnetModel, self).__init__()
        self.nblocks = nblocks

        
        self.ff_node = FeedForwardBloc([27, n_hidden_1, n_hidden_2])
        self.ff_edge = nn.Sequential(nn.Linear(27, n_hidden_1), 
                                     nn.Softplus(), 
                                     LinearBn(n_hidden_1, n_hidden_2))
        self.ff_state = FeedForwardBloc([2, n_hidden_1, n_hidden_2])
        
        megnet_bloc = []
        for i in range(nblocks):
            if i == 0:
                has_ff = False
            else:
                has_ff = True
            megnet_bloc.append(MegnetBloc(n_hidden_1, n_hidden_2, has_ff))
        self.megnet_bloc = nn.Sequential(*megnet_bloc)

        self.out_mlp = torch.nn.ModuleList([
            OutputLayer(n_hidden_2*4, y_mean[i], y_std[i]) for i in range(8)
        ])
    def forward(self, node, edge, state, index_1, index_2, gnode, gbond, coupling_index):
        node = self.ff_node(node)
        edge = self.ff_edge(edge)
        state = self.ff_state(state)
        node, edge, state, index_1, index_2, gnode, gbond = self.megnet_bloc([node, edge, state, index_1, 
                                                                              index_2, gnode, gbond])
        

        coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index = \
            torch.split(coupling_index,1,dim=1)
        node_0 = torch.index_select(node, dim = 0, index = coupling_atom0_index.view(-1))
        node_1 = torch.index_select(node, dim = 0, index = coupling_atom1_index.view(-1))
        
        coupling_atom_index = torch.cat([coupling_atom1_index, coupling_atom0_index], dim=-1)
        index_all = torch.cat([index_1.unsqueeze(1), index_2.unsqueeze(1)], dim=-1)

        indexs = self.get_edge_index(index_all, coupling_atom_index)
        edge = torch.index_select(edge, dim = 0, index = indexs)
        state = torch.index_select(state, dim = 0, index = coupling_batch_index.view(-1))

        final_vec = torch.cat([node_0, node_1, edge, state], dim = -1)

        pred = torch.zeros_like(coupling_type_index, dtype=torch.float, device=final_vec.device)
        for type_i in range(8):
            if (coupling_type_index == type_i).any():
                type_index = (coupling_type_index == type_i).nonzero()[:, 0]
                pred[coupling_type_index == type_i] = self.out_mlp[type_i](torch.index_select(final_vec, dim=0, index= type_index)).view(-1)

        return pred.view(-1)

    def get_edge_index(self, index_1, index_2):
        d = index_1.unsqueeze(0) - index_2.unsqueeze(1)
        dsum = torch.abs(d).sum(-1)
        indx = (dsum == 0).nonzero()[:, -1]
        return indx

    
class MegnetModel2(nn.Module):
    
    def __init__(self, 
                 nblocks = 5, 
                 n_hidden_1 = 512, 
                 n_hidden_2 =256, 
                 n_hidden_3 =128, 
                 ntarget=8):
                 
        super(MegnetModel, self).__init__()
        self.nblocks = nblocks

        
        self.ff_node = FeedForwardBloc([27, n_hidden_1, n_hidden_2])
        self.ff_edge = nn.Sequential(nn.Linear(27, n_hidden_1), 
                                     nn.Softplus(), 
                                     LinearBn(n_hidden_1, n_hidden_2))
        self.ff_state = FeedForwardBloc([2, n_hidden_1, n_hidden_2])
        
        megnet_bloc = []
        for i in range(nblocks):
            if i == 0:
                has_ff = False
            else:
                has_ff = True
            megnet_bloc.append(MegnetBloc(n_hidden_1, n_hidden_2, has_ff))
        self.megnet_bloc = nn.Sequential(*megnet_bloc)

        self.last_ff =  nn.Sequential(LinearBn(n_hidden_2*4, n_hidden_2, act = nn.Softplus()),
                                      LinearBn(n_hidden_2, n_hidden_3, act= nn.Softplus()),
                                      LinearBn(n_hidden_3, ntarget))


    def forward(self, node, edge, state, index_1, index_2, gnode, gbond, coupling_index):
        node = self.ff_node(node)
        edge = self.ff_edge(edge)
        state = self.ff_state(state)
        node, edge, state, index_1, index_2, gnode, gbond = self.megnet_bloc([node, edge, state, index_1, 
                                                                              index_2, gnode, gbond])
        

        coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index = \
            torch.split(coupling_index,1,dim=1)
        node_0 = torch.index_select(node, dim = 0, index = coupling_atom0_index.view(-1))
        node_1 = torch.index_select(node, dim = 0, index = coupling_atom1_index.view(-1))
        
        coupling_atom_index = torch.cat([coupling_atom1_index, coupling_atom0_index], dim=-1)
        index_all = torch.cat([index_1.unsqueeze(1), index_2.unsqueeze(1)], dim=-1)

        indexs = self.get_edge_index(index_all, coupling_atom_index)
        edge = torch.index_select(edge, dim = 0, index = indexs)
        state = torch.index_select(state, dim = 0, index = coupling_batch_index.view(-1))

        final_vec = torch.cat([node_0, node_1, edge, state], dim = -1)
        out = self.last_ff(final_vec)
        out = torch.gather(out, 1, coupling_type_index).view(-1)

        return out

    def get_edge_index(self, index_1, index_2):
        d = index_1.unsqueeze(0) - index_2.unsqueeze(1)
        dsum = torch.abs(d).sum(-1)
        indx = (dsum == 0).nonzero()[:, -1]
        return indx


