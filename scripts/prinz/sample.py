from argparse import ArgumentParser

import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import json

from bopito.data import data_utils
from bopito.data.prinz import BatchStochasticLaggedPrinzDataset
from bopito.models.ddpm import TensorDDPM
from bopito.models import score_models
from bopito.utils.mlops import get_checkpoint_config


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    paths = ["storage/results", "storage/results/prinz_potential", "storage/results/prinz_potential/ito", f"storage/results/prinz_potential/ito/{args.run_id}"]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
    
    with open(f"artifacts/model-{args.run_id}/config.json") as f:
        config = json.load(f)
    if args.model_type == 'bopito':
        checkpoint_path = get_checkpoint_config(project="bopito_prinz", run_id=args.run_id, tag="best")
        checkpoint_path_bg = get_checkpoint_config(project="bg_prinz", run_id=config["bg_id"], tag="best")
        sm_bg = score_models.BGScore(diffusion_steps=config["diffusion_steps"],
                                emb_dim=config["emb_dim"], net_dim = config["net_dim"]).to(device)
        tddpm_bg = TensorDDPM(
            sm_bg, diffusion_steps=config["diffusion_steps"], lr=config["lr"]
        ).to(device)
        tddpm_bg.load_state_dict(torch.load(checkpoint_path_bg))
        tddpm_bg.to(device)
        if args.use_different_lambda:
            sm = score_models.BoPITOScore(bg_score_model = tddpm_bg.score_model, max_lag=config["max_lag"],
                            emb_dim=config["emb_dim"], net_dim=config["net_dim"], max_eigenvalue= config["max_eigenvalue"], lambda_int =args.interpolation_lambda,
                            ).to(device)
        else:
            sm = score_models.BoPITOScore(bg_score_model = tddpm_bg.score_model, max_lag=config["max_lag"],
                                emb_dim=config["emb_dim"], net_dim=config["net_dim"], max_eigenvalue=config["max_eigenvalue"]).to(device)
    elif args.model_type == 'ito':
        checkpoint_path = get_checkpoint_config(project="ito_prinz", run_id=args.run_id, tag="best")
        sm = score_models.ITOScore(max_lag=config["max_lag"], diffusion_steps=config["diffusion_steps"],
                                emb_dim=config["emb_dim"], net_dim=config["net_dim"]).to(device)
    tddpm = TensorDDPM(
        sm, diffusion_steps=config["diffusion_steps"], lr=config["lr"]
    ).to(device)
    tddpm.load_state_dict(torch.load(checkpoint_path))
    tddpm.to(device)
    
    step_lag = tddpm.score_model.max_lag
    nested_sampling = args.nested_sampling
    if args.model_type == "bopito":
        if args.use_different_lambda:
            interpolation_lambda = args.interpolation_lambda
            if interpolation_lambda != tddpm.score_model.max_eigenvalue:
                print("Warning: interpolation_lambda is not equal to the max_eigenvalue of the model")
        else:
            interpolation_lambda = tddpm.score_model.max_eigenvalue
    
    initial_condition = args.initial_condition 

    for lag in args.lags:
        dataset = BatchStochasticLaggedPrinzDataset(max_lag=lag, fixed_lag=True)
        init_batch = data_utils.get_sample_batch_1d(dataset, args.batch_size)
        init_batch["corr"] = torch.randn_like(init_batch["target"], device=tddpm.device)
        if args.eq_init_cond:
            init_batch["cond"] = torch.ones_like(init_batch["target"], device=tddpm.device)*torch.tensor(np.load("storage/data/prinz/trajs.npy")[:args.batch_size,999], device=tddpm.device)
            init_batch["cond"] = init_batch["cond"].type(torch.float32)
        else:
            init_batch["cond"] = torch.ones_like(init_batch["target"], device=tddpm.device)*initial_condition
        init_batch["lag"] = init_batch["lag"].to(device)
        print(init_batch["corr"].device, init_batch["cond"].device, tddpm.device)
        if lag>config["max_lag"]:
            if args.model_type == 'bopito' and not nested_sampling:
                samples = tddpm.sample(init_batch, ode_steps=0)
            else:
                samples = tddpm.sample_nested(init_batch, lag, step_lag=step_lag, ode_steps=0)
        else:
            samples = tddpm.sample(init_batch, ode_steps=0)
        samples = samples.cpu().numpy()
        if not os.path.exists(f"storage/results/prinz_potential/{args.model_type}/{args.run_id}"):
            os.mkdir(f"storage/results/prinz_potential/{args.model_type}/{args.run_id}")
        if args.eq_init_cond:
            file_name = f'storage/results/prinz_potential/{args.model_type}/{args.run_id}/lag_{str(lag).zfill(3)}_ic_eq'
        else:
            file_name = f'storage/results/prinz_potential/{args.model_type}/{args.run_id}/lag_{str(lag).zfill(3)}_ic_{str(initial_condition)[:5]}'
        if args.use_different_lambda and args.model_type == 'bopito':
            file_name += f'_lambda_{str(interpolation_lambda)[:5]}'
        np.save(file_name+".npy", samples)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("run_id", type=str)
    parser.add_argument("--model_type",      type=str, default='ito')
    parser.add_argument("--lags",  nargs='+', default=[10, 25, 50, 75, 100, 200, 300, 500, 750, 1000])#
    parser.add_argument("--no_eq_init_cond", dest="eq_init_cond", action="store_false")
    parser.add_argument("--initial_condition",  type=float, default=0.75)
    parser.add_argument("--batch_size",  type=int, default=50000)
    parser.add_argument("--nested_sampling",     action='store_true')
    parser.add_argument("--use_different_lambda",     action='store_true')
    parser.add_argument("--interpolation_lambda",  type=float, default=0.99)
    
    # parser.add_argument('arg')
    # parser.add_argument('--kwarg')
    args = parser.parse_args()
    args.lags = [int(i) for i in args.lags]
    main(args)
