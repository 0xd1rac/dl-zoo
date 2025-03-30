import torch
from torch_geometric.data import Data

x_features = torch.tensor([[-1], [0], [1]], dtype=torch.long)
edge_index = torch.tensor([[0, 1, 1, 2]
                           [1, 0, 2, 1]
                           ], dtype=torch.float)

data = Data(x=x_features, edge_index=edge_index)
print(data)