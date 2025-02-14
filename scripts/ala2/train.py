import json
import os
import random
from argparse import ArgumentParser
from datetime import timedelta
from pprint import pprint

import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader

import wandb
from bopito import utils, DEVICE
from bopito.data.ala2 import StochasticLaggedALA2
from bopito.models import cpainn, ddpm
from bopito.utils import mlops

print("Imports done ...")


def main(args):  # pylint: disable=redefined-outer-name
    pprint(vars(args))
    #os.environ["WANDB_SILENT"] = "true"

    if args.seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.mps.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    job_dir = os.environ.get("JOB_DIR", None)
    if job_dir is None:
        job_dir = mlops.get_results_path(os.path.basename(__file__).split(".")[0])
    print(f"Job directory: {job_dir}")
    print("Loading data...", flush=True)
    
    dataset = get_dataset(args)

    print("Creating dataloader ...", flush=True)
    loader = GeometricDataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        train_time_interval=timedelta(minutes=args.save_freq),
    )

    if hasattr(dataset, "max_lag"):
        max_lag = dataset.max_lag
    else:
        max_lag = args.max_lag

    print("Instantiating model...", flush=True)
    if args.model_type == "bg":
        score_model = cpainn.PaiNNScore(
            score_layers=args.score_layers,
            n_features=args.n_features,
        )
    elif args.model_type == "ito":
        score_model = cpainn.PaiNNTLScore(
            max_lag=max_lag,
            embedding_layers=args.embedding_layers,
            score_layers=args.score_layers,
            n_features=args.n_features,
        )
    elif args.model_type == "bopito":
        bg_ckpt = mlops.get_checkpoint(args.project, args.bg_model)
        eq_score_model = ddpm.GeometricDDPM.load_from_checkpoint(checkpoint_path=bg_ckpt).score_model
        tl_score_model = cpainn.PaiNNTLScore(
            max_lag=max_lag,
            embedding_layers=args.embedding_layers,
            score_layers=args.score_layers,
            n_features=args.n_features,
        )
        score_model = cpainn.PaiNNBoPITOScore(eq_score_model, tl_score_model, args.lambda_hat)
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")

    model = ddpm.GeometricDDPM(
        score_model,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        alpha_bar_weight=args.alpha_bar_weight,
        #dont_evaluate=args.unconditional,
    )
    logger, run = mlops.get_wandb_logger(args.project, group=args.group, config=vars(args))

    with open(os.path.join(job_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)
    wandb.log(vars(args))

    print("Starting training...", flush=True)

    if args.model_id:
        model = mlops.get_model(args.model_id, args.model_tag, ddpm.GeometricDDPM)
        
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        enable_progress_bar=args.progress_bar,
        gradient_clip_val=0.5,
        accelerator=DEVICE,
        logger=logger,
        max_steps=args.max_steps,
        profiler=args.profiler,
        strategy="ddp_find_unused_parameters_true",
    )
    trainer.fit(model, loader)

    # utils.robust_training(trainer, model, loader, checkpoint_callback)


def get_dataset(args):  # pylint: disable=redefined-outer-name
    if args.data == "ala2":
        kwargs = {
            "max_lag": args.max_lag,
            "normalize": True,
            "n_traj": args.n_traj,
            "traj_indices": args.traj_indices,
            "path": args.datapath,
        }
        dataset = StochasticLaggedALA2(**kwargs)

    else:
        raise NotImplementedError(f"Dataset {args.data} not implemented")

    return dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    DEFAULT_WORKERS = 0
    if torch.cuda.is_available():
        DEFAULT_WORKERS = 0

    # fmt: off
    parser.add_argument("--data",             type=str, default='ala2')
    parser.add_argument("--datapath",         type=str,  default=None)
    parser.add_argument("--project",          type=str,  default="bopito-ala2")
    parser.add_argument("--split",            type=str,  default='train')
    parser.add_argument("--n_molecules",      type=int,  default=0)
    parser.add_argument("--max_lag",          type=int,  default=10000)
    parser.add_argument("--batch_size",       type=int, default=1024)
    parser.add_argument("--diffusion_steps",  type=int, default=1000)
    parser.add_argument("--n_features",       type=int, default=64)
    parser.add_argument("--score_layers",     type=int, default=5)
    parser.add_argument("--group",            type=str)
    parser.add_argument("--tags",             nargs="+", default=[])
    parser.add_argument("--embedding_layers", type=int, default=2)
    parser.add_argument("--fixed_lag",        action='store_true')
    parser.add_argument("--seed",             type=int, default=None)
    parser.add_argument("--progress_bar",     action='store_true')
    parser.add_argument("--num_workers",      type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--max_epochs",       type=int, default=100)
    parser.add_argument("--distinguish",      action='store_true')
    parser.add_argument("--test",             action='store_true')
    parser.add_argument("--molecules",        nargs='+', type=int, default= [])
    parser.add_argument("--alpha_bar_weight", action='store_true')
    parser.add_argument("--noise_schedule",   type=str, default="polynomial")
    parser.add_argument("--model_id",         type=str, default="")
    parser.add_argument("--model_tag",        type=str, default="best")
    parser.add_argument("--model_type",       type=str, default="ito")
    parser.add_argument("--bg_model",         type=str, default="None")
    parser.add_argument("--lambda_hat",       type=float, default=0.9996)
    parser.add_argument("--save_freq",        type=int, default=1)
    parser.add_argument("--max_steps",        type=int, default=-1)
    parser.add_argument('--profiler',         type=str, default='simple')
    parser.add_argument("--n_traj", type=lambda x: int(x) if x.isdigit() else None, default=None)
    parser.add_argument('--traj_indices',    nargs='+', default=[])
    # fmt: on

    args = parser.parse_args()
    args.traj_indices = [int(i) for i in args.traj_indices]
    assert not (
        args.n_molecules > 0 and args.molecule_idxs
    ), "Cannot specify both --molecule-idxs and --molecule-idxs-file"

    main(args)
