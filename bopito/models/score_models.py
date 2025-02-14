import torch
from torch import nn
import time

from bopito.models import embedding, layers


class ITOScore(nn.Module):
    def __init__(self, max_lag, diffusion_steps, xrange=None, emb_dim=32, net_dim=32):
        super().__init__()
        if xrange is None:
            xrange = [-6, 6]
        self.xrange = xrange
        self.max_lag = max_lag
        self.state_embedding = embedding.PositionalEncoder(dim=emb_dim, length=xrange[-1]-xrange[0], xmin=xrange[0])
        self.diffusion_time_embedding = embedding.PositionalEncoder(
            dim=emb_dim, length=diffusion_steps
        )
        self.physical_time_embedding = embedding.PositionalEncoder(
            dim=emb_dim, length=max_lag
        )
        self.net = layers.MLP(
            input_dim=4 * emb_dim, output_dim=1, nlayers=3, layer_size=net_dim
        )

    def forward(self, batch):
        t_phys = batch["lag"]
        t_phys_capped =  t_phys.clone()
        t_phys_capped[t_phys_capped > self.max_lag] = self.max_lag
        t_diff = batch["t_diff"]
        corr = batch["corr"]
        cond = batch["cond"].to(corr.device)

        t_phys_emb = self.physical_time_embedding(t_phys_capped).to(corr.device)
        t_diff_emb = self.diffusion_time_embedding(t_diff).to(corr.device)
        corr_emb = self.state_embedding(corr)
        cond_emb = self.state_embedding(cond)
        
        x = torch.cat((corr_emb, cond_emb, t_phys_emb, t_diff_emb), axis=1)
        
        out = self.net(x.squeeze(dim=-1))

        return out


class BGScore(nn.Module):
    def __init__(self, diffusion_steps, xrange=None, emb_dim=64, net_dim = 64):
        super().__init__()
        if xrange is None:
            xrange = [-6, 6]
        self.xrange = xrange
        self.state_embedding = embedding.PositionalEncoder(dim=emb_dim, length=xrange[-1]-xrange[0], xmin=xrange[0])
        self.diffusion_time_embedding = embedding.PositionalEncoder(
            dim=emb_dim, length=diffusion_steps
        )
        self.net = layers.MLP(
            input_dim=2*emb_dim, output_dim=1, nlayers=3, layer_size=net_dim
        )

    def forward(self, batch):
        t_diff = batch["t_diff"]
        corr = batch["corr"]
        t_diff_emb = self.diffusion_time_embedding(t_diff).to(corr.device)
        corr_emb = self.state_embedding(corr)
        x = torch.cat((corr_emb, t_diff_emb), axis=1)

        out = self.net(x.squeeze(dim=-1))
        return out

class BoPITOScore(nn.Module):
    def __init__(self, bg_score_model, max_lag, max_eigenvalue = 0.99, lambda_int =None, xrange=None, emb_dim=32, net_dim = 32): #slowest eigenvalue: 9.88078611e-01
        super().__init__()
        if xrange is None:
            xrange = [-6, 6]
        self.xrange = xrange
        self.max_lag = max_lag
        self.lambda_int = lambda_int
        self.state_embedding = embedding.PositionalEncoder(dim=emb_dim, length=xrange[-1]-xrange[0], xmin=xrange[0])
        self.diffusion_time_embedding = embedding.PositionalEncoder(
            dim=emb_dim, length=bg_score_model.diffusion_time_embedding.xmax
        )
        self.physical_time_embedding = embedding.PositionalEncoder(
            dim=emb_dim, length=max_lag
        )
        self.net = layers.MLP(
            input_dim=4*emb_dim, output_dim=1, nlayers=3, layer_size=net_dim
        )
        for param in bg_score_model.parameters(): param.requires_grad = False
        self.bg_score_model = bg_score_model  
        
        self.max_eigenvalue = torch.tensor(max_eigenvalue).to(next(bg_score_model.net.parameters()).device)

    def forward(self, batch):
        t_phys = batch["lag"]
        t_phys_capped =  t_phys.clone()
        t_phys_capped[t_phys_capped > self.max_lag] = self.max_lag
        
        t_diff = batch["t_diff"]
        corr = batch["corr"]
        cond = batch["cond"].to(corr.device)

        t_phys_emb = self.physical_time_embedding(t_phys_capped).to(corr.device)
        t_diff_emb = self.diffusion_time_embedding(t_diff).to(corr.device)
        corr_emb = self.state_embedding(corr)
        cond_emb = self.state_embedding(cond)
        x_bg = torch.cat((corr_emb, t_diff_emb), axis=1)
        x = torch.cat((corr_emb, cond_emb, t_phys_emb, t_diff_emb), axis=1)
        
        lambda_eff = torch.zeros(t_phys.shape, device=corr.device)
        if self.lambda_int != None:
            lambda_eff[t_phys <= self.max_lag] = self.max_eigenvalue
            lambda_eff[t_phys > self.max_lag] = self.lambda_int
        else:
            lambda_eff = self.max_eigenvalue
        return self.bg_score_model.net(x_bg.squeeze(dim=-1)) + lambda_eff**t_phys*self.net(x.squeeze(dim=-1))
    

