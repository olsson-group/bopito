import warnings

import torch
from torch_scatter import scatter

from bopito.models import embedding


class PaiNNTLScore(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
        self,
        n_features=32,
        embedding_layers=2,
        score_layers=5,
        max_lag=1000,
        diff_steps=1000,
        n_types=167,
        dist_encoding="positional_encoding",
    ):
        super().__init__()
        self.max_lag = max_lag
        layers = [
            embedding.AddSpatialEdgeFeatures(),
            embedding.NominalEmbedding(
                "bonds", n_features, n_types=4, feature_type="edge"
            ),
            embedding.NominalEmbedding("atoms", n_features, n_types=n_types),
            embedding.AddEquivariantFeatures(n_features),
            #  embedding.CombineInvariantFeatures(2 * n_features, n_features),
            PaiNNBase(
                n_features=n_features,
                n_features_out=n_features,
                n_layers=embedding_layers,
                dist_encoding=dist_encoding,
            ),
        ]

        self.embed = torch.nn.Sequential(*layers)

        self.net = torch.nn.Sequential(
            embedding.AddSpatialEdgeFeatures(),
            embedding.PositionalEmbedding("ts_diff", n_features, diff_steps),
            embedding.PositionalEmbedding("lag", n_features, max_lag),
            embedding.CombineInvariantFeatures(3 * n_features, n_features),
            PaiNNBase(
                n_features=n_features,
                dist_encoding=dist_encoding,
                n_layers=score_layers,
            ),
        )

    def forward(self, batch):
        cond = batch["cond"].clone().to(self.device)
        corr = batch["corr"].clone().to(self.device)

        batch_idx = batch["cond"].batch
        corr.lag = batch["lag"][batch_idx].squeeze()
        corr.ts_diff = batch["ts_diff"][batch_idx].squeeze()

        embedded = self.embed(cond)
        corr.invariant_node_features = embedded.invariant_node_features
        corr.equivariant_node_features = embedded.equivariant_node_features
        corr.invariant_edge_features = embedded.invariant_edge_features
        corr.edge_index = embedded.edge_index

        dx = self.net(corr).equivariant_node_features.squeeze()

        corr.x += dx
        return corr


class PaiNNBase(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
        self,
        n_features=128,
        n_layers=5,
        n_features_out=1,
        length_scale=10,
        dist_encoding="positional_encoding",
        use_edge_features=True,
    ):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(
                Message(
                    n_features=n_features,
                    length_scale=length_scale,
                    dist_encoding=dist_encoding,
                    use_edge_features=use_edge_features,
                )
            )
            layers.append(Update(n_features))

        layers.append(Readout(n_features, n_features_out))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, batch):
        return self.layers(batch)


class Message(torch.nn.Module):
    def __init__(
        self,
        n_features=128,
        length_scale=10,
        dist_encoding="positional_encoding",
        use_edge_features=True,
    ):
        super().__init__()
        self.n_features = n_features
        self.use_edge_features = use_edge_features

        assert dist_encoding in (
            a := ["positional_encoding", "soft_one_hot"]
        ), f"positional_encoder must be one of {a}"

        if dist_encoding in ["positional_encoding", None]:
            self.positional_encoder = embedding.PositionalEncoder(
                n_features, length=length_scale
            )
        elif dist_encoding == "soft_one_hot":
            self.positional_encoder = embedding.SoftOneHotEncoder(
                n_features, max_radius=length_scale
            )

        phi_in_features = 2 * n_features if use_edge_features else n_features
        self.phi = embedding.MLP(phi_in_features, n_features, 5 * n_features)
        self.w = embedding.MLP(n_features, n_features, 5 * n_features)

    def forward(self, batch):
        src_node = batch.edge_index[0]
        dst_node = batch.edge_index[1]

        in_features = batch.invariant_node_features[src_node]

        if self.use_edge_features:
            in_features = torch.cat(
                [in_features, batch.invariant_edge_features], dim=-1
            )

        positional_encoding = self.positional_encoder(batch.edge_dist)

        gates, cross_product_gates, scale_edge_dir, ds, de = torch.split(
            self.phi(in_features) * self.w(positional_encoding),
            self.n_features,
            dim=-1,
        )
        gated_features = multiply_first_dim(
            gates, batch.equivariant_node_features[src_node]
        )
        scaled_edge_dir = multiply_first_dim(
            scale_edge_dir, batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        )
        
        dst_node_edges = batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        dst_equivariant_node_features = batch.equivariant_node_features[dst_node]
        cross_produts = torch.cross(
            dst_node_edges, dst_equivariant_node_features, dim=-1
        )

        gated_cross_products = multiply_first_dim(cross_product_gates, cross_produts)

        dv = scaled_edge_dir + gated_features + gated_cross_products
        dv = scatter(dv, dst_node, dim=0)
        ds = scatter(ds, dst_node, dim=0)

        batch.equivariant_node_features += dv
        batch.invariant_node_features += ds
        batch.invariant_edge_features += de

        return batch


def multiply_first_dim(w, x):
    with warnings.catch_warnings(record=True):
        return (w.T * x.T).T


class Update(torch.nn.Module):
    def __init__(self, n_features=128):
        super().__init__()
        self.u = EquivariantLinear(n_features, n_features)
        self.v = EquivariantLinear(n_features, n_features)
        self.n_features = n_features
        self.mlp = embedding.MLP(2 * n_features, n_features, 3 * n_features)

    def forward(self, batch):
        v = batch.equivariant_node_features
        s = batch.invariant_node_features

        vv = self.v(v)
        uv = self.u(v)

        vv_norm = vv.norm(dim=-1)
        vv_squared_norm = vv_norm**2

        mlp_in = torch.cat([vv_norm, s], dim=-1)

        gates, scale_squared_norm, add_invariant_features = torch.split(
            self.mlp(mlp_in), self.n_features, dim=-1
        )

        delta_v = multiply_first_dim(uv, gates)
        delta_s = vv_squared_norm * scale_squared_norm + add_invariant_features

        batch.invariant_node_features = batch.invariant_node_features + delta_s
        batch.equivariant_node_features = batch.equivariant_node_features + delta_v

        return batch


class EquivariantLinear(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out):
        super().__init__()
        self.linear = torch.nn.Linear(n_features_in, n_features_out, bias=False)

    def forward(self, x):
        return self.linear(x.swapaxes(-1, -2)).swapaxes(-1, -2)


class Readout(torch.nn.Module):
    def __init__(self, n_features=128, n_features_out=13):
        super().__init__()
        self.mlp = embedding.MLP(n_features, n_features, 2 * n_features_out)
        self.V = EquivariantLinear(  # pylint:disable=invalid-name
            n_features, n_features_out
        )
        self.n_features_out = n_features_out

    def forward(self, batch):
        invariant_node_features_out, gates = torch.split(
            self.mlp(batch.invariant_node_features), self.n_features_out, dim=-1
        )

        equivariant_node_features = self.V(batch.equivariant_node_features)
        equivariant_node_features_out = multiply_first_dim(
            equivariant_node_features, gates
        )

        batch.invariant_node_features = invariant_node_features_out
        batch.equivariant_node_features = equivariant_node_features_out
        return batch


class PaiNNScore(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
        self,
        n_features=32,
        score_layers=3,
        diff_steps=1000,
        n_types=167,
        dist_encoding="positional_encoding",
    ):
        super().__init__()
        layers = [
            embedding.AddSpatialEdgeFeatures(),
            embedding.NominalEmbedding(
                "bonds", n_features, n_types=4, feature_type="edge"
            ),
            embedding.NominalEmbedding("atoms", n_features, n_types=n_types),
            embedding.PositionalEmbedding("ts_diff", n_features, diff_steps),
            embedding.AddEquivariantFeatures(n_features),
            embedding.CombineInvariantFeatures(2 * n_features, n_features),
            PaiNNBase(
                n_features=n_features,
                n_features_out=1,
                n_layers=score_layers,
                dist_encoding=dist_encoding,
            ),
        ]

        self.net = torch.nn.Sequential(*layers)

    def forward(self, batch):
        corr = batch["corr"].clone().to(self.device)
        batch_idx = batch["corr"].batch
        corr.ts_diff = batch["ts_diff"][batch_idx].squeeze()

        dx = self.net(corr).equivariant_node_features.squeeze()
        corr.x += dx

        return corr

class PaiNNBoPITOScore(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
        self,
        eq_score_model, tl_score_model, lambda_hat=0.9996
    ):
        super().__init__()
        for param in eq_score_model.parameters(): param.requires_grad = False
        self.eq_score_model = eq_score_model
        self.max_lag = tl_score_model.max_lag
        self.tl_score_model = tl_score_model
        self.lambda_hat = torch.tensor(lambda_hat, dtype=torch.double).to(next(eq_score_model.parameters()).device)
        
    def forward(self, batch):
        n_atoms = int(batch["cond"].x.size(0)/len(batch["lag"]))
        t_phys = batch["lag"]
        t_phys_capped =  t_phys.clone()
        t_phys_capped[t_phys_capped > self.max_lag] = self.max_lag
        t_phys_batch = torch.stack([t.repeat(n_atoms) for t in t_phys])
        t_phys_batch = t_phys_batch.view((-1,1))
        batch_copy = batch.copy()
        batch_copy["lag"] = t_phys_capped

        
        corr = batch["corr"].clone().to(self.device)
        pre_factor = self.lambda_hat**t_phys_batch
        corr.x = self.eq_score_model(batch_copy).x + pre_factor*self.tl_score_model(batch_copy).x

        return corr