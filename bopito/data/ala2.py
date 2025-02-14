import os
import requests
import tarfile

import mdshare
import mdtraj as md
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data as GeometricData

from bopito import DEVICE, utils
from bopito.data.datasets import StandardDatasetMixin, StochasticLaggedDatasetMixin

# fmt: off
ALA2ATOMNUMBERS = [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1]
SCALING_FACTOR = 1 / 0.1661689
# fmt: on


class ALA2Base:
    scaling_factor = SCALING_FACTOR

    def __init__(self, path=None, n_traj=None, traj_indices=[], normalize=False, distinguish=False):
        self.normalize = normalize
        self.atom_numbers = get_ala2_atom_numbers(distinguish)
        trajs = get_ala2_trajs(path)
        if traj_indices != []:
            trajs = [trajs[i] for i in traj_indices]
        elif n_traj is not None:
            trajs = trajs[:n_traj]
        traj_lens = []

        mol = Chem.MolFromPDBFile(
            "storage/data/ala2/alanine-dipeptide-nowater.pdb", removeHs=False
        )
        self.edge_index, self.bonds = utils.get_edge_attr_and_edge_index(mol)

        print("Loading trajectories...", flush=True)
        for traj in trajs:
            traj_lens.append(len(traj))

        self.trajs = np.concatenate(trajs)
        self.traj_boundaries = np.append([0], np.cumsum(traj_lens))

        if self.normalize:
            self.trajs *= self.scaling_factor

    def __getitem__(self, index):
        x = self.trajs[index]

        return GeometricData(
            x=torch.Tensor(x),
            atoms=self.atom_numbers,
            edge_index=self.edge_index,
            bonds=self.bonds,
        )


class ALA2(StandardDatasetMixin, ALA2Base):
    def __init__(self, path=None, normalize=False, distinguish=False):
        ALA2Base.__init__(self, path=path, normalize=normalize, distinguish=distinguish)
        StandardDatasetMixin.__init__(self)


class StochasticLaggedALA2(StochasticLaggedDatasetMixin, ALA2Base):
    def __init__(
        self, path=None, normalize=False, distinguish=False,
        n_traj = None, traj_indices = None, max_lag=1, fixed_lag=False
    ):
        ALA2Base.__init__(self, path, n_traj=n_traj, traj_indices=traj_indices, 
                          normalize=normalize, distinguish=distinguish)
        StochasticLaggedDatasetMixin.__init__(self, max_lag, fixed_lag)


def get_ala2_trajs(path=None):
    #check if files are there and if not download
    if os.path.exists("storage/data/ala2/alanine-dipeptide-nowater.pdb") and os.path.exists("storage/data/ala2/trajs.npy"):
        print("Loading alanine-dipeptide dataset ...")
    else:
        print("Downloading alanine-dipeptide dataset ...")
        download_ala2_trajs()
    
    topology = md.load("storage/data/ala2/alanine-dipeptide-nowater.pdb").topology
    if path is None:
        trajs = np.load("storage/data/ala2/trajs.npy")
    else:
        trajs = np.load(path)
    

    if np.inf in trajs:
        unpadded_trajs = []
        for traj in trajs:
            unpadded_trajs.append(traj[traj!=np.inf].reshape((-1, 22,3)))
        trajs = unpadded_trajs
    
    trajs = [md.Trajectory(t, topology) for t in trajs]
    trajs = [t.center_coordinates().xyz for t in trajs]

    return trajs


def download_ala2_trajs():
    tgz_path = 'storage/data/ala2/Ala2TSF300.tgz'
    np_path = 'storage/data/ala2/Ala2TSF300.npy'
    path = "storage/data/ala2"
    if not os.path.exists(np_path):
        print(f"Downloading alanine-dipeptide dataset to {path} ...")
        if not os.path.exists(path):
            os.makedirs(path)
        url = "http://ftp.mi.fu-berlin.de/pub/cmb-data/bgmol/datasets/ala2/Ala2TSF300.tgz"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(tgz_path, 'wb') as f:
                f.write(response.raw.read())
        
        file = tarfile.open(tgz_path)
        file.extractall(path)        
        trajs = np.load(np_path)
        trajs = trajs.reshape((20, 50000, 22, 3))
        np.save("storage/data/ala2/trajs.npy", trajs)

        filename = "alanine-dipeptide-nowater.pdb"    
        
        mdshare.fetch(filename, working_directory=path)
        
    return path


def get_ala2_atom_numbers(distinguish=False):
    atom_numbers = torch.tensor(  # pylint: disable=not-callable
        list(range(len(ALA2ATOMNUMBERS))) if distinguish else ALA2ATOMNUMBERS,
        device=DEVICE,
    )
    return atom_numbers
