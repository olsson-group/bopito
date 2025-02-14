import itertools

import mdtraj as md
import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as GeometricDataLoader

from bopito import utils
from bopito.data.ala2 import ALA2


def get_ala2_batch(batch_size, cond_sample_index=0, eq_init_cond=True, job_id=None):
    ds = ALA2()
    if eq_init_cond:
        if job_id is None:
            init_indices = np.load(f"storage/data/ala2/indices/eq_indices_00.npy")
        else:
            init_indices = np.load(f"storage/data/ala2/indices/eq_indices_{job_id.zfill(2)}.npy")
        batch_list = []
        for i in init_indices[:batch_size]:
            batch_list.append(ds[int(i)]["target"])
        return Batch.from_data_list(batch_list)
        #dl = GeometricDataLoader(ds, batch_size=batch_size)
        #return next(iter(dl))["target"]
    else:
        batch_list = []
        for _ in range(batch_size):
            batch_list.append(ds[cond_sample_index]["target"])
        return Batch.from_data_list(batch_list)


def get_atoms(mol):
    return [a.GetAtomicNum() for a in mol.GetAtoms()]
