import torch
from torch.nn import Linear, ReLU, Sequential, LayerNorm
from torch import nn
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import MetaLayer
from torch.functional import F


def linear_block(input_dim, output_dim, normalization=None, activation=None, dropout=None):
    layers = []
    if normalization:
        layers.append(normalization)
    layers.append(Linear(input_dim, output_dim))
    if activation:
        layers.append(activation)
    if dropout:
        layers.append(dropout)
    return layers


def create_mlp_v2(input_dim, output_dim, hidden_dims,
                  normalization_cls=nn.LayerNorm, activation_cls=nn.ReLU,
                  dropout_cls=nn.Dropout, dropout_prob=0.):
    layers = []
    in_ = input_dim
    for out_ in hidden_dims:
        layers.extend(
            linear_block(
                in_, out_,
                normalization=normalization_cls(
                    in_) if normalization_cls else None,
                activation=activation_cls() if activation_cls else None,
                dropout=dropout_cls(dropout_prob) if dropout_cls else None)
        )
        in_ = out_  # Next layer input_size
    # Output without activation and dropout
    layers.extend(
        linear_block(
            in_, output_dim,
            normalization=normalization_cls(
                in_) if normalization_cls else None,
            activation=None,
            dropout=None)
    )
    return Sequential(*layers)


def create_mlp(dims, input_dim, layer_norm=False):
    layers = []
    if layer_norm:
        layers.append(LayerNorm([input_dim]))
    layers.append(Linear(input_dim, dims[0]))
    for i in range(len(dims) - 1):
        layers.append(ReLU())
        if layer_norm:
            layers.append(LayerNorm([dims[i]]))
        layers.append(Linear(dims[i], dims[i+1]))
    return Sequential(*layers)


class MegNetLayer(torch.nn.Module):
    def __init__(self, dim=32, layer_norm=False):
        super(MegNetLayer, self).__init__()
        self.dim = dim
        self.edge_mlp = create_mlp(
            [dim * 2, dim * 2, dim], input_dim=dim * 4, layer_norm=layer_norm)
        self.node_mlp = create_mlp(
            [dim * 2, dim * 2, dim], input_dim=dim * 3, layer_norm=layer_norm)
        self.global_mlp = create_mlp(
            [dim * 2, dim * 2, dim], input_dim=dim * 3, layer_norm=layer_norm)

        def edge_model(src, dest, edge_attr, u, batch):
            # source, target: [E, F_x], where E is the number of edges.
            # edge_attr: [E, F_e]
            # u: [B, F_u], where B is the number of graphs.
            # batch: [E] with max entry B - 1.
            out = torch.cat([src, dest, edge_attr, u[batch]], 1)
            out = self.edge_mlp(out)
            return out

        def node_model(x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            row, _ = edge_index
            out = scatter_mean(edge_attr, row, dim=0, dim_size=x.size(0))
            out = torch.cat([out, x, u[batch]], dim=1)
            out = self.node_mlp(out)
            return out

        def global_model(x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            row, _ = edge_index
            edge_mean = scatter_mean(edge_attr, batch[row], dim=0)
            out = torch.cat(
                [u, scatter_mean(x, batch, dim=0), edge_mean], dim=1)
            out = self.global_mlp(out)
            return out

        self.op = MetaLayer(edge_model, node_model, global_model)

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.op(x, edge_index, edge_attr, u, batch)


class MegNetBlock(torch.nn.Module):
    def __init__(self, edge_dim, x_dim, u_dim, dim=32, residual=True, n_blocks=3, tied_weights=False):
        super(MegNetBlock, self).__init__()
        self.dim = dim
        self.residual = residual
        self.n_blocks = n_blocks

        self.edge_dense_first = create_mlp(
            [dim * 2, dim], input_dim=edge_dim, layer_norm=False)
        self.node_dense_first = create_mlp(
            [dim * 2, dim], input_dim=x_dim, layer_norm=False)
        self.global_dense_first = create_mlp(
            [dim * 2, dim], input_dim=u_dim, layer_norm=False)

        module_list = torch.nn.ModuleList
        if tied_weights:
            self.megnet = module_list([MegNetLayer(dim=self.dim)] * n_blocks)
            self.edge_dense = module_list([create_mlp([dim * 2, dim], input_dim=dim, layer_norm=True)]
                                          * (n_blocks - 1))
            self.node_dense = module_list([create_mlp([dim * 2, dim], input_dim=dim, layer_norm=True)]
                                          * (n_blocks - 1))
            self.global_dense = module_list([create_mlp([dim * 2, dim], input_dim=dim, layer_norm=True)]
                                            * (n_blocks - 1))
        else:
            self.megnet = module_list(
                [MegNetLayer(dim=self.dim) for _ in range(n_blocks)])
            self.edge_dense = module_list([create_mlp([dim * 2, dim], input_dim=dim, layer_norm=True)
                                           for _ in range(n_blocks-1)])
            self.node_dense = module_list([create_mlp([dim * 2, dim], input_dim=dim, layer_norm=True)
                                           for _ in range(n_blocks-1)])
            self.global_dense = module_list([create_mlp([dim * 2, dim], input_dim=dim, layer_norm=True)
                                             for _ in range(n_blocks-1)])

    def forward(self, x, edge_index, edge_attr, u, batch):
        # first block
        edge_out = self.edge_dense_first(edge_attr)
        x_out = self.node_dense_first(x)
        u_out = self.global_dense_first(u)
        for i in range(self.n_blocks):
            if i != 0:
                edge_head = self.edge_dense[i-1](edge_out)
                x_head = self.node_dense[i-1](x_out)
                u_head = self.global_dense[i-1](u_out)
            else:
                edge_head, x_head, u_head = edge_out, x_out, u_out

            x_head, edge_head, u_head = self.megnet[i](
                x_head, edge_index, edge_head, u_head, batch)

            if self.residual:
                edge_out = edge_out + edge_head
                x_out = x_out + x_head
                u_out = u_out + u_head
            else:
                edge_out = edge_head
                x_out = x_head
                u_out = u_head
        return x_out, edge_out, u_out


class DummyModel(torch.nn.Module):
    def __init__(self,
                 edge_dim,
                 x_dim,
                 u_dim,
                 dim=32,
                 n_megnet_blocks=3,
                 y_mean=0,
                 y_std=1):
        super(DummyModel, self).__init__()
        self.scaling = torch.nn.Linear(1, 1)
        self.scaling.bias = torch.nn.Parameter(torch.tensor(0,
                                                            dtype=torch.float),
                                               requires_grad=False)
        self.scaling.weight = torch.nn.Parameter(torch.tensor(
            [[1]], dtype=torch.float),
            requires_grad=True)

    def forward(self, data):
        return self.scaling(data.y)


class MegNetBlock_v2(torch.nn.Module):
    def __init__(self, edge_dim, x_dim, u_dim, dim=32, residual=True, n_blocks=3, tied_weights=False, layer_norm=False, inner_residual=False):
        super(MegNetBlock_v2, self).__init__()
        self.dim = dim
        self.residual = residual
        self.n_blocks = n_blocks

        self.edge_dense_first = create_mlp(
            [dim * 2, dim], input_dim=edge_dim, layer_norm=False)
        self.node_dense_first = create_mlp(
            [dim * 2, dim], input_dim=x_dim, layer_norm=False)
        self.global_dense_first = create_mlp(
            [dim * 2, dim], input_dim=u_dim, layer_norm=False)

        module_list = torch.nn.ModuleList
        if tied_weights:
            self.megnet = module_list([MegNetLayer_v2(
                dim=self.dim, layer_norm=layer_norm, residual=inner_residual)] * n_blocks)
            self.edge_dense = module_list([create_mlp([dim * 2, dim], input_dim=dim, layer_norm=layer_norm)]
                                          * (n_blocks - 1))
            self.node_dense = module_list([create_mlp([dim * 2, dim], input_dim=dim, layer_norm=layer_norm)]
                                          * (n_blocks - 1))
            self.global_dense = module_list([create_mlp([dim * 2, dim], input_dim=dim, layer_norm=layer_norm)]
                                            * (n_blocks - 1))
        else:
            self.megnet = module_list(
                [MegNetLayer_v2(dim=self.dim, layer_norm=layer_norm, residual=inner_residual) for _ in range(n_blocks)])
            self.edge_dense = module_list([create_mlp([dim * 2, dim], input_dim=dim, layer_norm=layer_norm)
                                           for _ in range(n_blocks-1)])
            self.node_dense = module_list([create_mlp([dim * 2, dim], input_dim=dim, layer_norm=layer_norm)
                                           for _ in range(n_blocks-1)])
            self.global_dense = module_list([create_mlp([dim * 2, dim], input_dim=dim, layer_norm=layer_norm)
                                             for _ in range(n_blocks-1)])

    def forward(self, x, edge_index, edge_attr, u, batch):
        # first block
        edge_out = self.edge_dense_first(edge_attr)
        x_out = self.node_dense_first(x)
        u_out = self.global_dense_first(u)
        for i in range(self.n_blocks):
            if i == 0:
                edge_head, x_head, u_head = edge_out, x_out, u_out
            else:
                edge_head = self.edge_dense[i-1](edge_out)
                x_head = self.node_dense[i-1](x_out)
                u_head = self.global_dense[i-1](u_out)

            x_head, edge_head, u_head = self.megnet[i](
                x_head, edge_index, edge_head, u_head, batch)

            if self.residual:
                x_out = x_out + x_head
                edge_out = edge_out + edge_head
                u_out = u_out + u_head
            else:
                edge_out, x_out, u_out = edge_head, x_head, u_head
        return x_out, edge_out, u_out


class MegNetLayer_v2(torch.nn.Module):
    def __init__(self, dim=32, layer_norm=False, residual=False):
        super(MegNetLayer_v2, self).__init__()
        self.dim = dim
        self.residual = residual
        self.edge_mlp = create_mlp(
            [dim * 2, dim * 2, dim], input_dim=dim * 4, layer_norm=layer_norm)
        self.node_mlp = create_mlp(
            [dim * 2, dim * 2, dim], input_dim=dim * 3, layer_norm=layer_norm)
        self.global_mlp = create_mlp(
            [dim * 2, dim * 2, dim], input_dim=dim * 3, layer_norm=layer_norm)

        def edge_model(src, dest, edge_attr, u, batch):
            # source, target: [E, F_x], where E is the number of edges.
            # edge_attr: [E, F_e]
            # u: [B, F_u], where B is the number of graphs.
            # batch: [E] with max entry B - 1.
            out = torch.cat([src, dest, edge_attr, u[batch]], 1)
            out = self.edge_mlp(out)
            if self.residual:
                return out + edge_attr
            else:
                return out

        def node_model(x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            row, _ = edge_index
            out = scatter_mean(edge_attr, row, dim=0, dim_size=x.size(0))
            out = torch.cat([out, x, u[batch]], dim=1)
            out = self.node_mlp(out)
            if self.residual:
                return out + x
            else:
                return out

        def global_model(x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            row, _ = edge_index
            edge_mean = scatter_mean(edge_attr, batch[row], dim=0)
            out = torch.cat(
                [u, scatter_mean(x, batch, dim=0), edge_mean], dim=1)
            out = self.global_mlp(out)
            if self.residual:
                return out + u
            else:
                return out

        self.op = MetaLayer(edge_model, node_model, global_model)

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.op(x, edge_index, edge_attr, u, batch)


POOLING_FN = {
    'mean': scatter_mean,
    'sum': scatter_add
}


class MegNetBlock_v3(torch.nn.Module):
    def __init__(self, edge_dim, x_dim, u_dim, dim=32, layer_norm=False,
                 normalization_cls=None, activation_cls=nn.ReLU,
                 dropout_cls=nn.Dropout, dropout_prob=0., residual=True, pooling='mean'):
        super(MegNetBlock_v3, self).__init__()
        self.dim = dim
        self.residual = residual
        self.pooling = pooling

        if layer_norm:
            normalization_cls = nn.LayerNorm
        kwargs = dict(
            normalization_cls=normalization_cls,
            activation_cls=activation_cls,
            dropout_cls=dropout_cls,
            dropout_prob=dropout_prob)
        self.edge_dense = create_mlp_v2(
            input_dim=edge_dim, output_dim=dim, hidden_dims=[dim * 2], **kwargs)
        self.node_dense = create_mlp_v2(
            input_dim=x_dim, output_dim=dim, hidden_dims=[dim * 2], **kwargs)
        self.global_dense = create_mlp_v2(
            input_dim=u_dim, output_dim=dim, hidden_dims=[dim * 2], **kwargs)

        self.edge_msg = create_mlp_v2(
            input_dim=dim * 4, output_dim=dim, hidden_dims=[dim*2, dim*2], **kwargs)
        self.node_msg = create_mlp_v2(
            input_dim=dim * 3, output_dim=dim, hidden_dims=[dim*2, dim*2], **kwargs)
        self.global_msg = create_mlp_v2(
            input_dim=dim * 3, output_dim=dim, hidden_dims=[dim*2, dim*2], **kwargs)

    def edge_model(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.edge_msg(out)
        return out

    def node_model(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, _ = edge_index
        out = POOLING_FN[self.pooling](
            edge_attr, row, dim=0, dim_size=x.size(0))
        out = torch.cat([out, x, u[batch]], dim=1)
        out = self.node_msg(out)
        return out

    def global_model(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, _ = edge_index
        edge_mean = scatter_mean(edge_attr, batch[row], dim=0)
        out = torch.cat(
            [u, scatter_mean(x, batch, dim=0), edge_mean], dim=1)
        out = self.global_msg(out)
        return out

    def forward(self, x, edge_index, edge_attr, u, batch, first_block=False):

        # first block
        edge_out = self.edge_dense(edge_attr)
        x_out = self.node_dense(x)
        u_out = self.global_dense(u)

        x_res_base = x_out if first_block else x
        edge_res_base = edge_out if first_block else edge_attr
        u_res_base = u_out if first_block else u

        row, col = edge_index

        edge_out = self.edge_model(x_out[row], x_out[col], edge_out, u_out,
                                   batch[row])
        if self.residual:
            edge_out = edge_res_base + edge_out

        x_out = self.node_model(x_out, edge_index, edge_out, u_out, batch)
        if self.residual:
            x_out = x_res_base + x_out

        u_out = self.global_model(x_out, edge_index, edge_out, u_out, batch)
        if self.residual:
            u_out = u_res_base + u_out

        return x_out, edge_out, u_out


class MegNetBlock_v4(torch.nn.Module):
    def __init__(self, edge_dim, x_dim, u_dim, dim=32, bottleneck_ratio=0.25,
                 normalization_cls=None, activation_cls=nn.ReLU,
                 dropout_cls=nn.Dropout, dropout_prob=0., residual=True, pooling='mean'):
        super(MegNetBlock_v4, self).__init__()
        self.dim = dim
        self.residual = residual
        self.pooling = pooling

        kwargs = dict(
            normalization_cls=normalization_cls,
            activation_cls=activation_cls,
            dropout_cls=dropout_cls,
            dropout_prob=dropout_prob)
        bottleneck_dim = int(dim * bottleneck_ratio)

        def one_layer_block(in_, out_):
            return Sequential(*linear_block(
                in_,
                out_,
                normalization=normalization_cls(
                    in_) if normalization_cls else None,
                activation=activation_cls() if activation_cls else None,
                dropout=dropout_cls(dropout_prob) if dropout_cls else None)
            )

        self.edge_dense = one_layer_block(edge_dim, bottleneck_dim)
        self.node_dense = one_layer_block(x_dim, bottleneck_dim)
        self.global_dense = one_layer_block(u_dim, bottleneck_dim)


        self.edge_msg = create_mlp_v2(
            input_dim=bottleneck_dim * 4, output_dim=bottleneck_dim, hidden_dims=[bottleneck_dim*2, bottleneck_dim*2], **kwargs)
        self.node_msg = create_mlp_v2(
            input_dim=bottleneck_dim * 3, output_dim=bottleneck_dim, hidden_dims=[bottleneck_dim*2, bottleneck_dim*2], **kwargs)
        self.global_msg = create_mlp_v2(
            input_dim=bottleneck_dim * 3, output_dim=bottleneck_dim, hidden_dims=[bottleneck_dim*2, bottleneck_dim*2], **kwargs)

        self.edge_out_dense = Sequential(*linear_block(
            bottleneck_dim,
            dim,
            normalization=normalization_cls(bottleneck_dim) if normalization_cls else None))
        self.node_out_dense = Sequential(*linear_block(
            bottleneck_dim,
            dim,
            normalization=normalization_cls(bottleneck_dim) if normalization_cls else None))
        self.global_out_dense = Sequential(*linear_block(
            bottleneck_dim,
            dim,
            normalization=normalization_cls(bottleneck_dim) if normalization_cls else None))

    def edge_model(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.edge_msg(out)
        return out

    def node_model(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, _ = edge_index
        out = POOLING_FN[self.pooling](
            edge_attr, row, dim=0, dim_size=x.size(0))
        out = torch.cat([out, x, u[batch]], dim=1)
        out = self.node_msg(out)
        return out

    def global_model(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, _ = edge_index
        edge_mean = scatter_mean(edge_attr, batch[row], dim=0)
        out = torch.cat(
            [u, scatter_mean(x, batch, dim=0), edge_mean], dim=1)
        out = self.global_msg(out)
        return out

    def forward(self, x, edge_index, edge_attr, u, batch):

        # first block
        edge_out = self.edge_dense(edge_attr)
        x_out = self.node_dense(x)
        u_out = self.global_dense(u)

        # conv
        row, col = edge_index
        edge_out = self.edge_model(x_out[row], x_out[col], edge_out, u_out,
                                   batch[row])
        x_out = self.node_model(x_out, edge_index, edge_out, u_out, batch)
        u_out = self.global_model(x_out, edge_index, edge_out, u_out, batch)

        # residual
        edge_out = edge_attr + self.edge_out_dense(edge_out)
        x_out = x + self.node_out_dense(x_out)
        u_out = u + self.global_out_dense(u_out)
        return x_out, edge_out, u_out
