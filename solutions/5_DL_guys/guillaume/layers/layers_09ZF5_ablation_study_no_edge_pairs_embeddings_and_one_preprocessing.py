from common import *
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.utils import softmax

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, max_hidden = 10000):
        super(BasicBlock, self).__init__()
        
        hidden_1 = min(max_hidden, (output_dim - input_dim) * 1 // 3 + input_dim)
        hidden_2 = min(max_hidden, (output_dim - input_dim) * 2 // 3 + input_dim)
      
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.bn1 = nn.LayerNorm(input_dim)
        
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.bn2 = nn.LayerNorm(hidden_1)
        
        self.fc3 = nn.Linear(hidden_2, output_dim)
        self.bn3 = nn.LayerNorm(hidden_2)
        
        self.relu = nn.Softplus()
        
    def forward(self, x):
        out = x
        
        out = self.bn1(out)
        out = self.fc1(out)
        
        out = self.relu(out)
        
        out = self.bn2(out)
        out = self.fc2(out)
        
        out = self.relu(out)
        
        out = self.bn3(out)
        out = self.fc3(out)
        
        return out

class AttentionAggregation(torch.nn.Module):

    def __init__(self, input_hidden, hidden):
        super(AttentionAggregation, self).__init__()

        self.input_hidden = input_hidden
        self.hidden = hidden

        self.proj_attention = BasicBlock(input_hidden, hidden)
        self.to_scalar = nn.Linear(hidden, 1)
        self.proj_values = BasicBlock(input_hidden, hidden)


    def forward(self, values, group, size):
        attention_embedding = torch.tanh(self.proj_attention(values))
        
        attention_embedding = self.to_scalar(attention_embedding)

        a = softmax(attention_embedding, group, num_nodes=size)
        r = scatter_add(a * self.proj_values(values), group, dim=0, dim_size=size)

        return r


class AttentionAggregation(torch.nn.Module):

    def __init__(self, input_hidden, hidden):
        super(AttentionAggregation, self).__init__()

        self.input_hidden = input_hidden
        self.hidden = hidden

        self.proj_attention = BasicBlock(input_hidden, hidden)
        self.to_scalar = nn.Linear(hidden, 1)
        self.proj_values = BasicBlock(input_hidden, hidden)


    def forward(self, values, group, size):
        attention_embedding = torch.tanh(self.proj_attention(values))
        
        attention_embedding = self.to_scalar(attention_embedding)

        r = scatter_add(attention_embedding * self.proj_values(values), group, dim=0, dim_size=size)

        return r


class AttentionAggregation(torch.nn.Module):

    def __init__(self, input_hidden, hidden):
        super(AttentionAggregation, self).__init__()

        self.input_hidden = input_hidden
        self.hidden = hidden

        self.proj_attention = nn.Linear(input_hidden, hidden)
        self.to_scalar = nn.Linear(hidden, 1)
        self.proj_values = nn.Linear(input_hidden, hidden)


    def forward(self, values, group, size):
        attention_embedding = torch.tanh(self.proj_attention(values))
        
        attention_embedding = self.to_scalar(attention_embedding)

        r = scatter_add(attention_embedding * self.proj_values(values), group, dim=0, dim_size=size)

        return r


class AttentionAggregation(torch.nn.Module):

    def __init__(self, input_hidden, hidden):
        super(AttentionAggregation, self).__init__()

        self.input_hidden = input_hidden
        self.hidden = hidden

        self.proj_attention = nn.Sequential(nn.LayerNorm(input_hidden), nn.Linear(input_hidden, input_hidden))
        self.normalize_attention = nn.LayerNorm(input_hidden)
        self.proj_values = nn.Linear(input_hidden, hidden)


    def forward(self, values, group, size):
        proj_values = self.proj_attention(values).unsqueeze(1)
        scalar = torch.bmm(proj_values, self.normalize_attention(values).unsqueeze(2))
        attention_embedding = torch.tanh(scalar).squeeze(2)

        r = scatter_add(attention_embedding * self.proj_values(values), group, dim=0, dim_size=size)

        return r

    

class AttentionAggregation(torch.nn.Module):

    def __init__(self, input_hidden, hidden):
        super(AttentionAggregation, self).__init__()

        self.input_hidden = input_hidden
        self.hidden = hidden

        self.proj_attention = nn.Sequential(nn.LayerNorm(input_hidden), nn.Linear(input_hidden, hidden), nn.Softplus(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.to_scalar = nn.Linear(hidden, 1)
        self.proj_values = nn.Sequential(nn.LayerNorm(input_hidden), nn.Linear(input_hidden, hidden), nn.Softplus(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden))


    def forward(self, values, group, size):
        attention_embedding = torch.tanh(self.proj_attention(values))
        
        attention_embedding = self.to_scalar(attention_embedding)

        r = scatter_add(attention_embedding * self.proj_values(values), group, dim=0, dim_size=size)

        return r

EDGE_TRIANGLE_SIZE = 50    

    
class MEGNetLayer(torch.nn.Module):
    def __init__(self, input_hidden, hidden):
        super().__init__()

        self.pre_edge_mlp = BasicBlock(input_hidden, hidden)
        self.pre_node_mlp = BasicBlock(input_hidden, hidden)
        self.pre_global_mlp = BasicBlock(input_hidden, hidden)
        
        self.edge_mlp_for_nodes = BasicBlock(hidden, hidden)
        self.edge_mlp_for_edges_and_global = BasicBlock(3 * hidden + EDGE_TRIANGLE_SIZE, hidden)
        
        #self.global_mlp = BasicBlock(3 * hidden + 2 * hidden, hidden)
        self.node_mlp = BasicBlock(3 * hidden, hidden)
        self.global_mlp = BasicBlock(3 * hidden, hidden)
        
        self.relu = nn.Softplus()
        
        self.agg_edge_1 = AttentionAggregation(3 * hidden, hidden)
        self.agg_edge_2 = AttentionAggregation(hidden, hidden)
        self.agg_node_1 = AttentionAggregation(hidden, hidden)
        
        self.agg_cycles_1 = AttentionAggregation(hidden, hidden)
        self.agg_cycles_2 = AttentionAggregation(hidden, hidden)
        
        
        
    def edge_model(self, src, dest, edge_attr, u, batch,
                   cycles_edge_index, cycles_id,
                   edges_connectivity_ids, transformed_edges_features):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        
        #"""
        if cycles_edge_index.size(0) != 0:
            cycles_embeddings = edge_attr[cycles_edge_index]
            cycles_embeddings_aggregated = self.agg_cycles_1(cycles_embeddings, cycles_id, cycles_id.max() + 1)
            cycles_embeddings_to_propagate = cycles_embeddings_aggregated[cycles_id]
            out_edge_embeddings_cycle_aggregated = self.agg_cycles_2(cycles_embeddings_to_propagate, cycles_edge_index, edge_attr.shape[0])
        else:
            out_edge_embeddings_cycle_aggregated = edge_attr.new_zeros(edge_attr.size())
        #"""
        
        
        edge_and_global = torch.cat([edge_attr, u[batch], out_edge_embeddings_cycle_aggregated, transformed_edges_features], dim = 1)
        out_edges_and_global = self.edge_mlp_for_edges_and_global(edge_and_global)
        out_nodes = self.edge_mlp_for_nodes(src) + self.edge_mlp_for_nodes(dest)
        out_classical = out_nodes + out_edges_and_global
        
        out_full = out_classical
        
        return out_full

    def node_model(self, x, row, col, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        num_node, node_dim = x.shape
        num_edge, edge_dim = edge_attr.shape

        edge = torch.cat([edge_attr, x[col], x[row]], dim = 1)
        #mean_edge = scatter_mean(edge, row, dim=0, dim_size=x.size(0))
        mean_edge = self.agg_edge_1(edge, row, x.size(0))
        out = torch.cat([mean_edge, x, u[batch]], dim=1)
        return self.node_mlp(out)

    def global_model(self, x, edge_attr, u, batch, edge_batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        
        #mean_nodes = scatter_mean(x, batch, dim=0)
        #mean_edges = scatter_mean(edge_attr, edge_batch, dim=0)
        
        mean_nodes = self.agg_node_1(x, batch, u.size(0))
        mean_edges = self.agg_edge_2(edge_attr, edge_batch, u.size(0))
        
        out = torch.cat([u, mean_nodes, mean_edges], dim=1)
        return self.global_mlp(out)

    def forward(self, x, edge_index, edge_attr, u, batch, edge_batch,
                cycles_edge_index,
                cycles_id,

                edges_connectivity_ids,
                transformed_edges_features):
        
        row, col = edge_index

        x_input, edge_attr_input, u_input = x, edge_attr, u
        
        x = self.pre_node_mlp(x)
        edge_attr = self.pre_edge_mlp(edge_attr)
        u = self.pre_global_mlp(u)
        
        # Edge model
        new_edge_attr = self.edge_model(x[row], x[col], edge_attr, u, edge_batch,
                                        cycles_edge_index, cycles_id,
                                        edges_connectivity_ids, transformed_edges_features)
        new_edge_attr += edge_attr_input

        row, col = torch.cat([row, col], dim = 0), torch.cat([col, row], dim = 0)
        double_new_edge_attr = torch.cat([new_edge_attr, new_edge_attr], dim = 0)
        
        # Node model
        new_x = self.node_model(x, row, col, double_new_edge_attr, u, batch)
        new_x += x_input

        # Global model
        new_u = self.global_model(new_x, new_edge_attr, u, batch, edge_batch)
        new_u += u_input

        return new_x, new_edge_attr, new_u
    
from torch_scatter import scatter_max, scatter_mean

class MEGNetList(torch.nn.Module):
    def __init__(self, count, nodes_embedding_in, edge_embedding_in, globals_embedding_in, nodes_in, edge_in, globals_in, 
                 hidden, target_means, target_stds):
        super().__init__()

        layers = []
        for i in range(count):
            layers.append(MEGNetLayer(hidden, hidden))
        self.op = nn.ModuleList(layers)
        
        """
        layers = []
        for i in range(count):
            layers.append(MEGNetLayer(hidden, hidden))
        self.op_sub = nn.ModuleList(layers)
        """
        
        input_description = [
            ['node', nodes_embedding_in, nodes_in],
            ['edge', edge_embedding_in, edge_in],
            ['global', globals_embedding_in, globals_in],
        ]
        
        preprocessing_dim_top = 700
        preprocessing = {}
        
        for name, embedding_input_size, numeric_input_size in input_description:
            current_preprocessing = {}
            
            current_preprocessing['numeric_projection'] = nn.ModuleList([nn.Linear(size, preprocessing_dim_top) for size, _ in numeric_input_size])
            current_preprocessing['embedding'] = nn.Embedding(embedding_input_size, preprocessing_dim_top)
            current_preprocessing['preprocessing'] = BasicBlock(preprocessing_dim_top, hidden)
            
            current_preprocessing = nn.ModuleDict(current_preprocessing)
            
            preprocessing[name] = current_preprocessing
            
        self.preprocessing = nn.ModuleDict(preprocessing)
        
        self.relu = nn.Softplus()

        self.projs = nn.ModuleList([
            nn.Sequential(BasicBlock(4 * hidden, 1), nn.Linear(1, 1)) for _ in range(8)
        ])
        
        
        for i in range(8):
            layer = self.projs[i][-1]
            layer.bias.requires_grad = False
            layer.bias[:] = target_means[i]
            layer.weight.requires_grad = False
            layer.weight[:] = target_stds[i] / 50


        self.transform_features_edges_connectivity = nn.ModuleList([nn.Sequential(
            nn.Linear(1, EDGE_TRIANGLE_SIZE),
            nn.Softplus(),
            nn.LayerNorm(EDGE_TRIANGLE_SIZE),
            nn.Linear(EDGE_TRIANGLE_SIZE, EDGE_TRIANGLE_SIZE)
        ) for _ in range(4)])
                                                                   
        self.transform_features_edges_connectivity_2 = nn.Sequential(
            nn.LayerNorm(4 * EDGE_TRIANGLE_SIZE),
            nn.Linear(4 * EDGE_TRIANGLE_SIZE, EDGE_TRIANGLE_SIZE),
            nn.Softplus(),
            nn.LayerNorm(EDGE_TRIANGLE_SIZE),
            nn.Linear(EDGE_TRIANGLE_SIZE, EDGE_TRIANGLE_SIZE)
        )
        
        self.agg_edge_connectivity_1 = AttentionAggregation(EDGE_TRIANGLE_SIZE, EDGE_TRIANGLE_SIZE)
            

    def forward(self,
                
                x_numeric,
                x_embeddings,
                 
                edge_attr_numeric, 
                edge_attr_embeddings,
     
                u_numeric, 
                u_embeddings, 
     
                edge_index,
                
                batch, 
                edge_batch, 
                
                target_types, 
                
                cycles_edge_index,
                cycles_id,

                edges_connectivity_ids,
                edges_connectivity_features):

        inputs = {
            'node' : (x_numeric, x_embeddings),
            'edge' : (edge_attr_numeric, edge_attr_embeddings),
            'global' : (u_numeric, u_embeddings),
        }
        
        for key in inputs:
            numeric_projection = torch.cat([proj(input_) for proj, input_ in zip(self.preprocessing[key]['numeric_projection'], inputs[key][0])], dim = 1)
            embedding = self.preprocessing[key]['embedding'](inputs[key][1]).sum(dim = 1)
            
            input_top = numeric_projection + embedding
            
            input_bottom = self.preprocessing[key]['preprocessing'](input_top)
            
            inputs[key] = input_bottom
        
        
        x = inputs['node']
        edge_attr = inputs['edge']
        u = inputs['global']
        
        
        # Preprocess edge_connectivity_features
        transformed_edges_features = torch.cat([
            self.transform_features_edges_connectivity[i](edges_connectivity_features[:, i].unsqueeze(1)) for i in range(4)
        ], dim = 1)                       
        transformed_edges_features = self.transform_features_edges_connectivity_2(transformed_edges_features)
        
        transformed_edges_features = self.agg_edge_connectivity_1(transformed_edges_features, edges_connectivity_ids[0], edge_attr.shape[0])
          
        
        for i in range(len(self.op)):
            x, edge_attr, u = self.op[i](x, edge_index, edge_attr, u, batch, edge_batch,
                                         
                                         cycles_edge_index,
                                         cycles_id,

                                         edges_connectivity_ids,
                                         transformed_edges_features)
           
        row, col = edge_index
        
        pool_u  = torch.index_select(u, dim=0, index=edge_batch)
        node0 = torch.index_select(x, dim=0, index=row)
        node1 = torch.index_select(x, dim=0, index=col)
        
        predict = torch.cat([node0, node1, edge_attr, pool_u],-1)
        
        predicts = [proj(predict) for proj in self.projs]
        predict = torch.cat(predicts, dim = 1)
        predict = torch.gather(predict,1,target_types)

        return predict
    
    