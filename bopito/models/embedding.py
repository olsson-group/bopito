import numpy as np
import torch

import numpy as np
import torch


class MLP(torch.nn.Module):
    def __init__(self, f_in, f_hidden, f_out, skip_connection=False):
        super().__init__()
        self.skip_connection = skip_connection

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(f_in, f_hidden),
            torch.nn.LayerNorm(f_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(f_hidden, f_hidden),
            torch.nn.LayerNorm(f_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(f_hidden, f_out),
        )

    def forward(self, x):
        if self.skip_connection:
            return x + self.mlp(x)

        return self.mlp(x)


class AddEdgeIndex(torch.nn.Module):
    def __init__(self, n_neighbors=None, cutoff=None):
        super().__init__()
        self.n_neighbors = n_neighbors if n_neighbors else 10000
        self.cutoff = cutoff if cutoff is not None else float("inf")

    def forward(self, batch):
        batch = batch.clone()
        edge_index = self.generate_edge_index(batch)
        batch.edge_index = edge_index.to(batch.x.device)
        return batch


class AddSpatialEdgeFeatures(torch.nn.Module):
    def forward(self, batch, *_, **__):
        r = batch.x[batch.edge_index[0]] - batch.x[batch.edge_index[1]]

        edge_dist = r.norm(dim=-1)
        edge_dir = r / (1 + edge_dist.unsqueeze(-1))

        batch.edge_dist = edge_dist
        batch.edge_dir = edge_dir
        return batch


class InvariantFeatures(torch.nn.Module):
    """
    Implement embedding in child class
    All features that will be embedded should be in the batch
    """

    def __init__(self, feature_name, type_="node"):
        super().__init__()
        self.feature_name = feature_name
        self.type = type_

    def forward(self, batch):
        embedded_features = self.embedding(batch[self.feature_name])

        name = f"invariant_{self.type}_features"
        if hasattr(batch, name):
            batch[name] = torch.cat([batch[name], embedded_features], dim=-1)
        else:
            batch[name] = embedded_features

        return batch


class NominalEmbedding(InvariantFeatures):
    def __init__(self, feature_name, n_features, n_types, feature_type="node"):
        super().__init__(feature_name, feature_type)
        self.embedding = torch.nn.Embedding(n_types, n_features)


class DeviceTracker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("device_tracker", torch.tensor(1))

    @property
    def device(self):
        return self.device_tracker.device
    
#class PositionalEncoder(torch.nn.Module):
#    def __init__(self, dim, xmax=10, xmin=0):
#        super().__init__()
#        assert dim % 2 == 0, "dim must be even for positional encoding for sin/cos"
#        self.dim = dim
#
#        self.xmin = xmin
#        self.xmax = xmax
#        self.length= xmax - xmin
#
#        self.max_rank = dim // 2
#        self.min = min

#    def forward(self, x):
#        return torch.concatenate(
#            [self.positional_encoding(x, rank) for rank in range(self.max_rank)],
#            axis=1,
#        )

#    def positional_encoding(self, x, rank):
#        sin = torch.sin((x - self.xmin) / self.length * rank * np.pi)
#        cos = torch.cos((x - self.xmin) / self.length * rank * np.pi)
#        return torch.concatenate((cos, sin), axis=1)



class PositionalEncoder(DeviceTracker):
    def __init__(self, dim, length=10, xmin=0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for positional encoding for sin/cos"

        self.dim = dim
        self.xmin = xmin
        self.length = length
        self.xmax = xmin + length
        self.max_rank = dim // 2

    def forward(self, x):
        
        encodings = [self.positional_encoding(x, rank) for rank in range(self.max_rank)]
        return torch.cat(
            encodings,
            axis=1,
        )

    def positional_encoding(self, x, rank):
        sin = torch.sin((x - self.xmin) / self.length * rank * np.pi)
        cos = torch.cos((x - self.xmin) / self.length * rank * np.pi)
        assert (
            cos.device == self.device
        ), f"batch device {cos.device} != model device {self.device}"
        return torch.stack((cos, sin), axis=1)


class PositionalEmbedding(InvariantFeatures):
    def __init__(self, feature_name, n_features, length):
        super().__init__(feature_name)
        assert n_features % 2 == 0, "n_features must be even"
        self.rank = n_features // 2
        self.embedding = PositionalEncoder(n_features, length)


class CombineInvariantFeatures(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out):
        super().__init__()
        self.mlp = MLP(n_features_in, n_features_out, n_features_out)

    def forward(self, batch):
        batch.invariant_node_features = self.mlp(batch.invariant_node_features)
        return batch


class AddEquivariantFeatures(DeviceTracker):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

    def forward(self, batch):
        eq_features = torch.zeros(
            batch.batch.shape[0],
            self.n_features,
            3,
        )
        batch.equivariant_node_features = eq_features.to(self.device)
        return batch



