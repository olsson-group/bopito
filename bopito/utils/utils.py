import subprocess
import webbrowser
import os

import mdtraj as md
import numpy as np
import ot
import requests
import torch
from mdtraj import element
from urllib3.exceptions import NewConnectionError

import wandb
from bopito import DEVICE
from bopito.utils import mlops
from bopito.models import ddpm, score_models

def is_jupyter_server_running(port):
    try:
        response = requests.get(f"http://localhost:{port}/api")
        if response.status_code == 200:
            return True
    except (requests.ConnectionError, NewConnectionError):
        return False
    return False


ELEMENTS = {
    1: element.hydrogen,
    6: element.carbon,
    7: element.nitrogen,
    8: element.oxygen,
    9: element.fluorine,
}


def batch_to_device(batch, device):
    for key in batch.keys():
        if hasattr(batch[key], "to"):
            batch[key] = batch[key].to(device)
        if isinstance(batch[key], dict):
            batch[key] = batch_to_device(batch[key], device)

    return batch


def get_topology(atom_numbers):
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("RES", chain)

    for i, atom_number in enumerate(atom_numbers):
        e = ELEMENTS[atom_number]
        name = f"{e}{i}"
        topology.add_atom(name, e, residue)

    return topology


def sample_to_pdb(sample_path):
    sample = md.load(sample_path)
    sample.save_pdb("/tmp/sample.pdb")


def save_traj(traj, atoms):
    traj = create_mdtraj(traj, atoms)
    traj.save("/tmp/traj.pdb")


def create_mdtraj(traj, atoms):
    topology = get_topology(atoms)
    traj = traj.reshape(-1, *traj.shape[-2:])
    traj = md.Trajectory(traj, topology)
    return traj


def robust_training(trainer, model, loader, checkpoint_callback):
    attempt = 0
    failed_step = 0

    while True:
        ckpt = lm_ckpt if (lm_ckpt := checkpoint_callback.best_model_path) else None

        if ckpt:
            model = type(model).load_from_checkpoint(checkpoint_path=ckpt)

        try:
            trainer.fit(model, loader, ckpt_path=ckpt)
            break

        except ValueError:
            timestamp = mlops.get_timestamp()
            wandb.log(
                {
                    "attempt": attempt,
                    "timestamp": timestamp,
                    "global_step": trainer.global_step,
                }
            )
            attempt += 1
            if trainer.global_step == failed_step:
                print(f"Failed twice at global step {failed_step}. Exiting.")
                __import__("sys").exit()
            if attempt >= 1000:
                print("Too many attempts. Exiting.")
                __import__("sys").exit()


def filter_trajs(trajs):
    mask = abs(trajs).max(axis=(1, 2, 3)) > 10
    mask += np.isnan(trajs).any(axis=(1, 2, 3))
    return trajs[~mask]


def geomdata_to_tensor(batch):
    poss = [data.x for data in batch.to_data_list()]
    return poss


def geomdata_to_numpy(batch):
    return geomdata_to_tensor(batch).cpu().numpy()


def get_dihedrals(traj, dihedral_atoms):
    mdtraj = create_mdtraj(traj, np.ones(traj.shape[-2]))
    dihedrals = md.compute_dihedrals(mdtraj, dihedral_atoms)
    return dihedrals.T


def wasserstein_distance_circle(d1, d2):
    try:
        normalized_d1 = (d1 + np.pi) / (2 * np.pi)
        normalized_d2 = (d2 + np.pi) / (2 * np.pi)
        normalized_d = ot.wasserstein_circle(normalized_d1, normalized_d2)[0]
        return normalized_d * 2 * np.pi
    except ZeroDivisionError:
        return None


def to_device(batch, device=None):
    if device is None:
        device = DEVICE
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        if isinstance(batch[key], dict):
            to_device(batch[key], device)


def get_edge_attr_and_edge_index(mol):
    bond_matrix = get_bond_matrix(mol)
    edge_index = torch.stack(torch.where(bond_matrix != -1))
    edge_attr = bond_matrix[edge_index[0], edge_index[1]]
    return edge_index, edge_attr


def get_bond_matrix(mol):
    N = mol.GetNumAtoms()
    bonds = get_bonds(mol)

    bond_matrix = torch.zeros((N, N), dtype=int)
    bond_matrix[bonds[:, 0], bonds[:, 1]] = bonds[:, 2]
    bond_matrix[bonds[:, 1], bonds[:, 0]] = bonds[:, 2]
    bond_matrix += torch.eye(N, dtype=torch.long) * -1

    return bond_matrix


def get_bonds(mol):
    bonds = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
        for bond in mol.GetBonds()
    ]
    return torch.tensor(bonds, dtype=torch.long)


def train_step_bg(batches, i, tddpm, optimizer, device, pbar=None): 
    batch = batches[i]
    batch["target"]= batch["target"].to(device) 
    loss = tddpm.training_step(batch, i)
    wandb.log({"train/loss": loss})
    tddpm.on_before_zero_grad()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if pbar is not None:   pbar.set_description(f"Loss {loss.detach().cpu().numpy()}")
    return loss

def train_step_dyn(batches, i, tddpm, optimizer, device, pbar=None):
    batch = batches[i]
    batch["target"]= batch["target"].to(device) 
    batch["cond"] = batch["cond"].to(device)
    batch["lag"] = batch["lag"].to(device)
    loss = tddpm.training_step(batch, i)
    wandb.log({"train/loss": loss})
    tddpm.on_before_zero_grad()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if pbar is not None:   pbar.set_description(f"Loss {loss.detach().cpu().numpy()}")
    return loss

def get_prinz_score_model(config, device):
    if config["model_type"] == "ito":
        sm = score_models.ITOScore(max_lag = config["max_lag"], diffusion_steps=config["diffusion_steps"], 
                                   emb_dim = config["emb_dim"], net_dim = config["net_dim"]).to(device)
    elif config["model_type"] == "bg":
        sm = score_models.BGScore(diffusion_steps=config["diffusion_steps"],
                              emb_dim=config["emb_dim"], net_dim = config["net_dim"]).to(device)
    elif config["model_type"] == "bopito":
        checkpoint_path = mlops.get_checkpoint_config(project="bg_prinz", run_id=config["bg_id"], tag="best")
        sm_bg = score_models.BGScore(diffusion_steps=config["diffusion_steps"],
                                emb_dim=config["emb_dim"], net_dim = config["net_dim"]).to(device)
        tddpm_bg = ddpm.TensorDDPM(
            sm_bg, diffusion_steps=config["diffusion_steps"], lr=config["lr"]
        )
        tddpm_bg.load_state_dict(torch.load(checkpoint_path))
        tddpm_bg.to(device)
        sm = score_models.BoPITOScore(bg_score_model = tddpm_bg.score_model, max_lag=config["max_lag"],
                                emb_dim=config["emb_dim"], net_dim=config["net_dim"], max_eigenvalue=config["max_eigenvalue"]).to(device)
    else:
        raise ValueError("Model type not recognized")
        
    return sm
    

def get_model_type(args):
    api = wandb.Api()
    run = api.run(args.project+f"/{args.run_id}")
    model_type = run.config.get("model_type")
    return model_type

def get_results_handler(args):
    if args.model_type == "ito" or args.model_type == "bopito":
        if args.eq_init_cond:
            if len(args.lag_schedule) > 1:
                path_name = os.path.join("cond_sample", args.run_id, "lags_"+str(args.lag_schedule[0]).zfill(5))
            else:
                path_name = os.path.join("cond_sample", args.run_id, "lag_"+str(args.lag).zfill(5))
        else:
            
            if len(args.lag_schedule) > 1:
                path_name = os.path.join("cond_sample", args.run_id, "lags_"+str(args.lag_schedule[0]).zfill(5)+"_id_"+str(args.sample_index))
            else:
                path_name = os.path.join("cond_sample", args.run_id, "lag_"+str(args.lag).zfill(5)+"_id_"+str(args.sample_index))
        if args.job_id is not None:
            results_handler = mlops.ResultsHandler(path_name, "job_"+args.job_id)
        else:
            results_handler = mlops.ResultsHandler(path_name)
    else:
        results_handler = mlops.ResultsHandler(
        os.path.join("bg_sample", args.run_id, args.molecule)
        )
        
    return results_handler

def retrieve_data(model, lag=0, bg=False, nested=False, n_jobs = 7, init = None, interpolation = False):
    sampled_positions = []
    if bg:
        models_dir = f"experiments/bg_sample/{model}/ala2/"
        for subdir in sorted(os.listdir(models_dir)):
            sub_dir_path = os.path.join(models_dir, subdir)
            if os.path.isdir(sub_dir_path):
                npy_file = os.path.join(sub_dir_path, "sample.npy")
                #print(npy_file)
                array = np.reshape(np.load(npy_file)*10, (-1,22,3))
                sampled_positions.append(array)
                #print(array.shape)
    else:
        if not interpolation:
            if not nested:
                if init is not None:
                    models_dir = f"experiments/cond_sample/{model}/lag_"+str(lag).zfill(5)+"_id_"+str(init)+"/"
                    #print(models_dir)
                else:
                    models_dir = f"experiments/cond_sample/{model}/lag_"+str(lag).zfill(5)+"/"
            else:
                if init is not None:
                    models_dir = f"experiments/cond_sample/{model}/lag_00500"+"_id_"+str(init)+"/"
                else:
                    models_dir = f"experiments/cond_sample/{model}/lag_00500/"
                #print(models_dir)
        else:
            if nested:
                if init is not None:
                    models_dir = f"experiments/cond_sample/{model}/lags_"+str(lag).zfill(5)+"_id_"+str(init)+"/"
                else:
                    models_dir = f"experiments/cond_sample/{model}/lags_"+str(lag).zfill(5)+"/"
            else:
                if init is not None:
                    models_dir = f"experiments/cond_sample/{model}/lag_"+str(lag).zfill(5)+"_id_"+str(init)+"/"
                    #print(models_dir)
                else:
                    models_dir = f"experiments/cond_sample/{model}/lag_"+str(lag).zfill(5)+"/"
        jobs= []
        for subdir in sorted(os.listdir(models_dir)):
            job = subdir[:5]
            sub_dir_path = os.path.join(models_dir, subdir)
            if os.path.exists(os.path.join(sub_dir_path, "sample.npy")) and job not in jobs:
                npy_file = os.path.join(sub_dir_path, "sample.npy")
                if nested:
                    array = np.load(npy_file)*10
                else:
                    array = np.load(npy_file)*10
                sampled_positions.append(array)
                jobs.append(job)
                
        if len(jobs) < n_jobs:
            print(f"Warning: Found less than {n_jobs} jobs for model {model} and lag {lag}.")
            print("Found jobs:", jobs)   
                
        if nested:
            return np.concatenate(sampled_positions, axis=0)

    return np.concatenate(sampled_positions, axis=0)