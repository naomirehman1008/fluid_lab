from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        layers = []
        in_dim = input_size
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, graph_lev=False):
        super().__init__()
        self.graph_lev = graph_lev

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_size, hidden_size))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_size, hidden_size))
        self.convs.append(GCNConv(hidden_size, output_size))

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)

        # if graph-level prediction, pool over nodes
        if self.graph_lev:
            x = global_mean_pool(x, batch=None)

        return x