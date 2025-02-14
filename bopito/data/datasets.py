import h5py
import numpy as np
import torch
from tqdm import tqdm

from bopito import DEVICE


class StochasticLaggedDatasetMixin:
    def __init__(self, max_lag, fixed_lag=False):
        self.max_lag = max_lag
        self.data0_idx = self.compute_data0_idxs(max_lag)
        self.fixed_lag = fixed_lag

    def compute_data0_idxs(self, max_lag):
        data0_idxs = []

        for start, end in zip(self.traj_boundaries[:-1], self.traj_boundaries[1:]):
            traj_len = end - start
            non_lagged_length = max(0, traj_len - max_lag-1)
            data0_idxs.extend(range(start, start + non_lagged_length))

        return np.array(data0_idxs)

    def __len__(self):
        return len(self.data0_idx)

    def __getitem__(self, idx):
    
        log_lag = np.random.uniform(0, np.log(self.max_lag+1))
        lag = int(np.floor(np.exp(log_lag)))
   

        if self.fixed_lag:
            lag = self.max_lag

        data0_idx = self.data0_idx[idx]
        data0 = super().__getitem__(data0_idx)
        datat = super().__getitem__(data0_idx + lag)
        
        return {'cond': data0, "target": datat, 'lag': torch.Tensor([lag])}


class StandardDatasetMixin:
    def __len__(self):
        return self.traj_boundaries[-1]

    def __getitem__(self, idx):
        data = super().__getitem__(idx).to(DEVICE)
        return {"target": data}


class LazyH5DatasetMixin:
    def __init__(self, path, lazy_load=False):
        self.path = path
        self.loaded = False
        if not lazy_load:
            self.lazy_load()

    def lazy_load(self):
        if not self.loaded:
            self.h5file = h5py.File(self.path, "r")
            self.loaded = True
    
class BatchStochasticLaggedDatasetMixin:
    def __init__(self, max_lag, fixed_lag=False):
        self.max_lag = max_lag
        self.data0_idx = self.compute_data0_idxs(max_lag)
        self.fixed_lag = fixed_lag

    def compute_data0_idxs(self, max_lag):
        data0_idxs = []

        for start, end in zip(self.traj_boundaries[:-1], self.traj_boundaries[1:]):
            traj_len = end - start
            non_lagged_length = max(0, traj_len - max_lag-1)
            data0_idxs.extend(range(start, start + non_lagged_length))

        return np.array(data0_idxs)

    def __len__(self):
        return len(self.data0_idx)

    def __getitem__(self, idx):
        if type(idx) != int:
            data0_idxs = self.data0_idx[idx]
            data0 = super().__getitem__(data0_idxs)
            log_lag = np.random.uniform(0, np.log(self.max_lag+1), len(idx))
            lag = np.floor(np.exp(log_lag))
            lag = lag.astype(int)
            if self.fixed_lag:
                lag = np.ones(len(idx)).astype(int)*self.max_lag
            datat = super().__getitem__(data0_idxs + lag)
            
            data0 = data0.view(len(idx),1)
            datat = datat.view(len(idx),1)
            lag = torch.Tensor(lag).view(len(idx),1)

            return {'cond': data0, "target": datat, 'lag': lag}
        else:
            log_lag = np.random.uniform(0, np.log(self.max_lag+1))
            lag = int(np.floor(np.exp(log_lag)))

            if self.fixed_lag:
                lag = self.max_lag

            data0_idx = self.data0_idx[idx]
            data0 = super().__getitem__(data0_idx)
            datat = super().__getitem__(data0_idx + lag)

            return {'cond': data0, "target": datat, 'lag': torch.Tensor([lag])}

class UniformBatchStochasticLaggedDatasetMixin:
    #Note: Assumes lenght of all trajectories are the same length.
    def __init__(self, max_lag, fixed_lag=False):
        self.max_lag = max_lag
        self.fixed_lag = fixed_lag
        self.data0_idx = self.compute_data0_idxs()
        self.traj_length = self.traj_boundaries[1]-self.traj_boundaries[0]
        
    def compute_data0_idxs(self):
        data0_idxs = []

        if self.fixed_lag:
            for start, end in zip(self.traj_boundaries[:-1], self.traj_boundaries[1:]):
                traj_len = end - start
                non_lagged_length = max(0, traj_len -self.max_lag- 1)
                data0_idxs.extend(range(start, start + non_lagged_length))
        else:
            for start, end in zip(self.traj_boundaries[:-1], self.traj_boundaries[1:]):
                traj_len = end - start
                non_lagged_length = max(0, traj_len -1)
                data0_idxs.extend(range(start, start + non_lagged_length))

        return np.array(data0_idxs)
    
    def __len__(self):
        return len(self.data0_idx)

    def __getitem__(self, idx):
        if type(idx) != int:
            data0_idxs = self.data0_idx[idx]
            data0 = super().__getitem__(data0_idxs)
            
            if self.fixed_lag:
                lag = np.ones(len(idx)).astype(int)*self.max_lag
            else:
                max_lag_traj = self.traj_boundaries[ np.array((data0_idxs/self.traj_length+1), dtype=int) ]-np.array(data0_idxs)-1
                max_lag_traj[max_lag_traj>self.max_lag] = self.max_lag
                lag = np.copy(max_lag_traj)
                
                lag[max_lag_traj!=1] = np.random.randint(1, max_lag_traj[max_lag_traj!=1]+1)
            
            datat = super().__getitem__(data0_idxs + lag)
            
            data0 = data0.view(len(idx),1)
            datat = datat.view(len(idx),1)
            lag = torch.Tensor(lag).view(len(idx),1)

            return {'cond': data0, "target": datat, 'lag': lag}
        else:
            data0 = super().__getitem__(idx)
            
            max_lag_traj = self.traj_boundaries[ int(idx/self.traj_length )]
            lag = np.random.randint(1, max_lag_traj+1-idx, len(idx))
            
            datat = super().__getitem__(idx + lag)

            return {'cond': data0, "target": datat, 'lag': torch.Tensor([lag])}

class StandardDatasetMixin:
    def __len__(self):
        return self.traj_boundaries[-1]

    def __getitem__(self, idx):
        data = super().__getitem__(idx).to(DEVICE)
        if type(idx) != int:
            return {"target": data.view(len(idx),1)}
        else:
            return {"target": data}
