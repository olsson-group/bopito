import json
import os
from argparse import ArgumentParser
import wandb

import torch
import numpy as np
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm

from bopito import DEVICE
from bopito.data import ala2
from bopito.data.ala2 import ALA2
from bopito.models import ddpm
from bopito.utils import mlops, sampling
from bopito.utils.utils import create_mdtraj, get_model_type, get_results_handler

DATASET_CLASSES = {
    "ala2": ALA2,
}

SCALING_FACTORS = {
    "ala2": ala2.SCALING_FACTOR,
    "none": 1.0,
}

def main(args):
    model_type = get_model_type(args)
    args.model_type = model_type
    results_handler = get_results_handler(args)

    with open(results_handler.get_path("args.json"), "w") as f:
        json.dump(vars(args), f)

    model = get_model(args.run_id, args.tag)
    
    if args.model_type == "ito" or args.model_type =="bopito":
    
        cond = get_init_batch(args)
        scaling_factor = SCALING_FACTORS[args.scaling]

        cond.x *= scaling_factor
        
        
        if len(args.lag_schedule) > 1:
            traj = sample_fn(
            model, cond, None, args.ode_steps, args.nested_samples, burnin=args.burnin, lag_schedule=args.lag_schedule,
            )
        else:

            traj = sample_fn(
                model, cond, args.lag, args.ode_steps, args.nested_samples, burnin=args.burnin
            )
        traj = traj.detach().cpu().numpy() / scaling_factor
        print(traj.shape)

        #sample = {
        #    "traj": traj,
        #    "lag": args.lag,
        #    "atoms": cond[0].atoms.detach().cpu().numpy(),
        #    "model": f"{args.project}_{args.run_id}_{args.tag}",
        #    "molecule_idx": str(args.molecule_idx).zfill(5),
        #}

    elif args.model_type == "bg":
        scaling_factor = SCALING_FACTORS[args.scaling]
        dataset = DATASET_CLASSES[args.molecule](normalize=False)
        loader = GeometricDataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        batch = next(iter(loader))
        conf = batch["target"]
        conf = model.sample_like(conf, ode_steps=args.ode_steps)
        conf = conf.x.detach().cpu().numpy() / scaling_factor
        conf = conf.reshape(args.batch_size, -1, 3)
        
        #sample = {
        #    "traj": conf,
        #    "atoms": batch["target"].atoms.detach().cpu().numpy(),
        #    "model": f"{args.project}_{args.run_id}_{args.tag}",
        #    "molecule_idx": str(args.molecule_idx).zfill(5),
        #}
        traj = conf
        
    #mdtraj = create_mdtraj(traj, sample["atoms"])
    np.save(results_handler.get_path("sample.npy"), traj)
    #mdtraj.save(results_handler.get_path("sample.pdb"))

    #results_handler.save(sample, "sample.pkl")
    print(f'Sample saved to {results_handler.get_path("sample.npy")}')
    results_handler.slurm()


def sample_fn(model, cond, lag,  ode_steps, nested_samples, lag_schedule=[], burnin=0):
    if lag is not None and len(lag_schedule)> 1:
        raise ValueError("Cannot provide both lag and lag_schedule.")    
    if len(lag_schedule)> 1:
            print(f"Using lag schedule {lag_schedule}...")
            nested_samples = len(lag_schedule)
    samples = [cond]

    for _ in tqdm(range(burnin)):
        sample = model.sample_cond(cond, lag=lag, ode_steps=ode_steps)
        cond = sample

    for i_nested in tqdm(range(nested_samples)):
        if len(lag_schedule)> 1:
            lag = lag_schedule[i_nested]
            print(f"Sampling with lag {lag}...")
        sample = model.sample_cond(cond, lag=lag, ode_steps=ode_steps)
        cond = sample
        samples.append(cond)

    x = group_trajs(samples)

    return x


def get_model(run_id, tag):
    ckpt = mlops.get_checkpoint("bopito-ala2", run_id, tag=tag)
    model = ddpm.GeometricDDPM.load_from_checkpoint(checkpoint_path=ckpt)
    print("Model loaded ...")
    return model


def get_init_batch(args):
    if args.molecule == "ala2":
        return sampling.get_ala2_batch(args.batch_size, job_id=args.job_id, eq_init_cond=args.eq_init_cond, cond_sample_index=args.sample_index).to(DEVICE)

    if args.batch:
        return mlops.load(args.batch)

    data = args.data if args.data else args.project
    kwargs = {}

    if args.molecule_idx is not None:
        kwargs["split"] = [str(args.molecule_idx).zfill(5)]

    dataset = DATASET_CLASSES[data](normalize=False, **kwargs)
    loader = GeometricDataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(loader))

    cond = batch["target"]
    cond.to(DEVICE)
    cond.x *= DATASET_CLASSES[args.project].scaling_factor

    return cond


def get_scaling_factor(args):
    data = args.data if args.data else args.project
    return DATASET_CLASSES[data].scaling_factor


def group_pos(sample):
    pos = torch.stack([conf.x for conf in sample.to_data_list()])
    return pos


def group_trajs(samples):
    trajs = torch.stack([group_pos(sample) for sample in samples], axis=1)
    return trajs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("run_id", type=str)
    parser.add_argument("--project", type=str, default="bopito-ala2")
    parser.add_argument("--lag", type=int, default=10)
    parser.add_argument("--ode_steps", type=int, default=50)
    parser.add_argument("--nested_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=7500)
    parser.add_argument("--molecule_idx", type=int, default=None)
    parser.add_argument("--tag", type=str, default="best")
    parser.add_argument("--molecule", type=str, default="ala2")
    parser.add_argument("--scaling", type=str, default="ala2")
    parser.add_argument("--burnin", type=int, default=0)
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--no_eq_init_cond", dest="eq_init_cond", action="store_false")
    parser.add_argument('--lag_schedule',    nargs='+', default=[])
    
    args = parser.parse_args()                        
    args.lag_schedule = [int(i) for i in args.lag_schedule]

    main(args)
