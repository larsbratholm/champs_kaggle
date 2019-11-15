import torch
from torch_scatter import scatter_max, scatter_add

class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=True)
        # self.bn   = nn.BatchNorm1d(out_channel)
        self.layer_norm = nn.LayerNorm(in_channel)
        self.act  = act

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear(x)
        if self.act is not None:
            x = self.act(x)
        # x = self.bn(x)
        return x
    
class FeedForwardBloc(nn.Module):
    def __init__(self,
                 channels):
        super(FeedForwardBloc, self).__init__()
        shapes = list(zip(channels[:-1], channels[1:]))
        mlp_list = [LinearBn(i, j, nn.ELU()) for (i, j) in shapes]
        self.mlp = nn.Sequential(*mlp_list)

    def forward(self, x):
        return self.mlp(x)

class EdgeAggregation(nn.Module):
    def __init__(self, input_hidden, hidden):
        super(EdgeAggregation, self).__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(input_hidden),
                                 nn.Linear(input_hidden, hidden),
                                 nn.ELU(), 
                                 nn.LayerNorm(hidden), 
                                 nn.Linear(hidden, hidden))
        self.gate = nn.Sequential(nn.Linear(hidden, 1),
                                  nn.Sigmoid())
        self.out = nn.Linear(hidden, input_hidden)
    
    def forward(self, edges, edges_index):
        out = self.mlp(edges)
        out = self.out(out) * self.gate(out)
        res = scatter_add(out, edges_index, dim=0)
        return res
    
def linear_block(in_dim, out_dim, normalization, 
                activation, dropout):
    layers = []
    layers.append(normalization)
    layers.append(nn.Linear(in_dim, out_dim))
    if activation: layers.append(activation)
    if dropout: layers.append(dropout)
    return nn.Sequential(*layers)

def create_mlp(input_dim, output_dim, hidden_dims,
               normalization_cls=nn.LayerNorm, activation_cls=nn.Softplus,
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
        in_ = out_ 

    layers.extend(
        linear_block(
            in_, output_dim,
            normalization=normalization_cls(
                in_) if normalization_cls else None,
            activation=None,
            dropout=None)
    )
    return nn.Sequential(*layers)

class OutputLayer(torch.nn.Module):
    def __init__(self, input_dim, y_mean, y_std):
        super(OutputLayer, self).__init__()
        self.scaling = torch.nn.Linear(1, 1)
        self.scaling.bias = torch.nn.Parameter(torch.tensor(y_mean,
                                                            dtype=torch.float),
                                               requires_grad=False)
        self.scaling.weight = torch.nn.Parameter(torch.tensor([[y_std]], 
                                                 dtype=torch.float),requires_grad=False)
        
        self.mlp = create_mlp(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=[input_dim//2, input_dim//4, input_dim//4],
            normalization_cls=torch.nn.LayerNorm,
            activation_cls=torch.nn.Softplus,
            dropout_cls=torch.nn.Dropout,
            dropout_prob=0.
        )

    def forward(self, data):
        out = self.mlp(data)
        out = self.scaling(out)
        return out