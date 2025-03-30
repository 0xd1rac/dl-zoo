import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_mean, scatter_max

class GraphSAGELayer(MessagePassing):
    def __init__(self, in_channels, out_channels, aggregator="mean"):
        super().__init__(aggr=None)  # Disable default aggregation
        self.aggregator = aggregator
        self.lin = nn.Linear(in_channels * 2, out_channels)

        if aggregator == "lstm":
            self.lstm = nn.LSTM(in_channels, in_channels, batch_first=True)
        elif aggregator == "maxpool":
            self.pool_lin = nn.Linear(in_channels, in_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # Return raw neighbor features; aggregation is done later
        if self.aggregator == "maxpool":
            return F.relu(self.pool_lin(x_j))
        return x_j

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Aggregates messages from neighbors per node.
        Args:
            inputs: messages (i.e., x_j or transformed x_j)
            index: target node indices for each message
        """
        if self.aggregator == "mean":
            return scatter_mean(inputs, index, dim=0, dim_size=dim_size)
        elif self.aggregator == "maxpool":
            return scatter_max(inputs, index, dim=0, dim_size=dim_size)[0]
        elif self.aggregator == "lstm":
            # Group messages by target node
            from torch_geometric.utils import to_dense_batch

            # Convert sparse to dense batch by node
            dense_inputs, mask = to_dense_batch(inputs, index)
            h_lstm, _ = self.lstm(dense_inputs)  # batch_size x max_neighbors x in_channels
            return h_lstm[:, -1, :]  # Use last output (or .mean(1), depending on your goal)

    def update(self, aggr_out, x):
        # Combine self features with aggregated neighbor info
        h = torch.cat([x, aggr_out], dim=1)
        return F.relu(self.lin(h))
