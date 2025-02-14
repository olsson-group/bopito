import torch
from torch import nn

from bopito.models import embedding


class MLP(nn.Module):
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        layer_size=128,
        nlayers=5,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, layer_size))
        layers.append(nn.SiLU())
        for _ in range(1, nlayers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(layer_size, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPScore(MLP):
    def __init__(
        self,
        diffusion_steps,
        nlayers=5,
        data_dim=1,
        layer_size=128,
        time_dim=32,
    ):
        super().__init__(
            input_dim=data_dim + time_dim,
            output_dim=data_dim,
            layer_size=layer_size,
            nlayers=nlayers,
        )

        self.diffusion_time_encoder = embedding.PositionalEncoder(
            time_dim, diffusion_steps
        )

    def forward(self, batch):  # pylint: disable=arguments-differ
        t_diff = batch["t_diff"]
        corr = batch["corr"]
        t_diff_emb = self.diffusion_time_encoder(t_diff)
        x_ = torch.cat((corr, t_diff_emb), axis=1)

        out = self.net(x_)
        return out


class ConditionalMLPScore(MLP):
    def __init__(
        self,
        diffusion_steps,
        #  max_lag,
        nlayers=5,
        data_dim=1,
        layer_size=128,
        time_dim=32,
        skip_connection=False,
    ):
        super().__init__(
            input_dim=2 * data_dim + time_dim,
            output_dim=data_dim,
            layer_size=layer_size,
            nlayers=nlayers,
        )
        self.diffusion_steps = diffusion_steps
        self.skip_connection = skip_connection
        self.diffusion_time_encoder = embedding.PositionalEncoder(
            time_dim, diffusion_steps
        )
        #  self.lag_time_encoder = embedding.PositionalEncoder(time_dim, max_lag)

    def forward(self, batch):
        t_diff = batch["t_diff"]
        x_0 = batch["cond"]
        x_t = batch["corr"]

        t_diff_emb = self.diffusion_time_encoder(t_diff)

        x_ = torch.cat((x_0, x_t, t_diff_emb), axis=1)

        if self.skip_connection:
            return self.net(x_) + x_0
        return self.net(x_)
