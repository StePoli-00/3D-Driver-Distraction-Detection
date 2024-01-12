import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool


class GATNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=6, dropout_prob=0.2):
        super(GATNet, self).__init__()

        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)

        self.readout = nn.Sequential(
                nn.Linear(hidden_dim * num_heads, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
        )
        self.dropout = nn.Dropout(dropout_prob)

        self.skip_conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)
        self.skip_conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply the skip connections
        x_skip = F.relu(self.skip_conv1(x, edge_index))
        x_skip = self.dropout(x_skip)
        x_skip = F.relu(self.skip_conv2(x_skip, edge_index))
        x_skip = self.dropout(x_skip)

        # Apply the main GAT layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        # Add the skip connections to the main GAT layers
        x += x_skip

        x_readout = global_mean_pool(x, batch)
        x_global = self.readout(x_readout)
        return x_global.squeeze(),x_readout
