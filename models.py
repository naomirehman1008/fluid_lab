import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from topomodelx.nn.simplicial.scconv import SCConv

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
            x = global_mean_pool(x, batch=batch)

        return x
    
class SCConvNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, complex_lev=True, n_layers=1):
        super().__init__()
        self.linear_x0_in = torch.nn.Linear(in_channels, hidden_channels)
        self.linear_x1_in = torch.nn.Linear(in_channels, hidden_channels)
        self.linear_x2_in = torch.nn.Linear(in_channels, hidden_channels)
        self.complex_lev = True

        self.base_model = SCConv(
            node_channels=hidden_channels,
            n_layers=n_layers,
        )

        self.linear_x0 = torch.nn.Linear(hidden_channels, out_channels)
        self.linear_x1 = torch.nn.Linear(hidden_channels, out_channels)
        self.linear_x2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        incidence_1,
        incidence_1_norm,
        incidence_2,
        incidence_2_norm,
        adjacency_up_0_norm,
        adjacency_up_1_norm,
        adjacency_down_1_norm,
        adjacency_down_2_norm,
    ):
        x_0 = self.linear_x0_in(x_0)
        x_1 = self.linear_x1_in(x_1)
        x_2 = self.linear_x2_in(x_2)

        x_0, x_1, x_2 = self.base_model(
            x_0,
            x_1,
            x_2,
            incidence_1,
            incidence_1_norm,
            incidence_2,
            incidence_2_norm,
            adjacency_up_0_norm,
            adjacency_up_1_norm,
            adjacency_down_1_norm,
            adjacency_down_2_norm,
        )

        x_0 = self.linear_x0(x_0)
        x_1 = self.linear_x1(x_1)
        x_2 = self.linear_x2(x_2)

        return global_mean_pool(x_0, batch=None)