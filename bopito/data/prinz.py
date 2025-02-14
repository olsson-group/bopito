import os
from argparse import ArgumentParser

from torch.utils.data import Dataset

import numpy as np
import torch
from deeptime import clustering, data
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM

from bopito.data import datasets


class PrinzDatasetBase:
    def __init__(self, path="storage/data/prinz/trajs.npy", n_traj = None, max_length = None, traj_indices = [], burn_in = None):
        trajs = get_trajs(path)
        if traj_indices != []:
            trajs = [trajs[int(i)] for i in traj_indices]
        else:
            if n_traj is not None:
                trajs = trajs[:n_traj]
        if max_length is not None:
            if burn_in is not None:
                trajs = np.array([traj[burn_in:burn_in+max_length] for traj in trajs])
            else:
                trajs = np.array([traj[:max_length] for traj in trajs]) 
        traj_lens = [len(traj) for traj in trajs]
        self.traj_boundaries = np.append([0], np.cumsum(traj_lens))
        #self.data = trajs.flatten()
        self.data = np.concatenate(trajs)
        
    def __len__(self):
        return self.traj_boundaries[-1]

    def __getitem__(self, idx):
        return torch.Tensor(self.data[None, idx])
    
class BGPrinzDatasetBase(Dataset):
    def __init__(self, path="storage/data/prinz/trajs.npy", device ="cuda"):
        trajs = get_trajs(path)
        traj_lens = [len(traj) for traj in trajs]
        self.traj_boundaries = np.append([0], np.cumsum(traj_lens))
        self.data = trajs.flatten()
        self.data = torch.Tensor(self.data).to(device)

    def __len__(self):
        return self.traj_boundaries[-1]

    def __getitem__(self, idx):
        return self.data[idx]


class StochasticLaggedPrinzDataset(
    datasets.StochasticLaggedDatasetMixin, PrinzDatasetBase
):
    def __init__(
        self, path="storage/data/prinz/trajs.npy", max_lag=1000, fixed_lag=False,
        n_traj = None, max_length = None
    ):
        PrinzDatasetBase.__init__(self, path, n_traj = None, max_length = None)
        datasets.StochasticLaggedDatasetMixin.__init__(self, max_lag, fixed_lag)
        
class BatchStochasticLaggedPrinzDataset(
    datasets.UniformBatchStochasticLaggedDatasetMixin, PrinzDatasetBase
):
    def __init__(
        self, path="storage/data/prinz/trajs.npy", max_lag=1000, fixed_lag=False,
        n_traj = None, max_length = None, traj_indices = [], burn_in = None
    ):
        PrinzDatasetBase.__init__(self, path, n_traj = n_traj, max_length = max_length, traj_indices = traj_indices, burn_in= burn_in)
        datasets.UniformBatchStochasticLaggedDatasetMixin.__init__(self, max_lag, fixed_lag=fixed_lag)
        
class BatchStochasticLogLaggedPrinzDataset(
    datasets.BatchStochasticLaggedDatasetMixin, PrinzDatasetBase
):
    def __init__(
        self, path="storage/data/prinz/trajs.npy", max_lag=1000, fixed_lag=False,
        n_traj = None, max_length = None, traj_indices = [], burn_in = None
    ):
        PrinzDatasetBase.__init__(self, path, n_traj = n_traj, max_length = max_length, traj_indices = traj_indices, burn_in= burn_in)
        datasets.BatchStochasticLaggedDatasetMixin.__init__(self, max_lag, fixed_lag)


class PrinzDataset(datasets.StandardDatasetMixin, PrinzDatasetBase):
    def __init__(self, path="storage/data/prinz/trajs.npy"):
        PrinzDatasetBase.__init__(self, path)
        datasets.StandardDatasetMixin.__init__(self)
        
class BGPrinzDataset(datasets.StandardDatasetMixin, BGPrinzDatasetBase):
    def __init__(self, path="storage/data/prinz/trajs.npy"):
        BGPrinzDatasetBase.__init__(self, path)
        datasets.StandardDatasetMixin.__init__(self)


def get_trajs(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        trajs = np.load(path)
    else:
        print("Data path does not exist...")
        print("Generating data, this will take about 10 minutes ...")
        trajs = generate_prinz_data(path)

    return trajs


def generate_prinz_data(path, n_trajs=50000, traj_len=2000, burnin=2000):
    prinz_potential = data.prinz_potential()  # pylint: disable=redefined-outer-name
    trajs = prinz_potential.trajectory(np.zeros((n_trajs, 1)), traj_len + burnin)
    trajs = trajs[:, burnin:]

    np.save(path, trajs)

    return trajs
