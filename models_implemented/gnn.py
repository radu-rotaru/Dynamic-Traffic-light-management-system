import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(nn.Module):
    def __init__(self, number_of_nodes, number_of_edges, number_of_node_features, hidden_layers_dims, output_layer_dim):
        super(GNN, self).__init__()
        self.number_of_nodes = number_of_nodes
        self.number_of_edges = number_of_edges
        self.hidden_layers_dims = hidden_layers_dims
        
        self.conv1 = GCNConv(number_of_node_features, hidden_layers_dims[0])
        self.conv2 = GCNConv(hidden_layers_dims[0], hidden_layers_dims[1])

    def forward(self, node_features, edges):
        edge_index = torch.tensor(edges, dtype=torch.long)
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = edge_index.t().contiguous()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x