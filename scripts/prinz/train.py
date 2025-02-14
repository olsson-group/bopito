from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from tqdm import tqdm
import wandb
import time
import os
import json

from bopito.data.prinz import BatchStochasticLaggedPrinzDataset, BatchStochasticLogLaggedPrinzDataset
from bopito.models import ddpm
from bopito.utils.mlops import get_wandb_logger
from bopito.data.data_utils import generate_batches
from bopito.utils.utils import train_step_dyn, get_prinz_score_model


def main(args):
    config = args.__dict__
    print(config)  
    
    model_type = config["model_type"]
    project = f"{model_type}_prinz"
    logger, run = get_wandb_logger(project, config=config)
    project_path = config["project_path"]
    
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if not os.path.exists(project_path+"/artifacts/"): os.mkdir(project_path+"/artifacts/")
    os.mkdir(project_path+f"/artifacts/model-{wandb.run.id}")
    with open(project_path+f"/artifacts/model-{wandb.run.id}/config.json", "w") as outfile: 
        json.dump(config, outfile)
    model_path = project_path+f"/artifacts/model-{wandb.run.id}/model.pth"

    sm = get_prinz_score_model(config, device)
    #

    tddpm = ddpm.TensorDDPM(
        sm, diffusion_steps=config["diffusion_steps"], lr=config["lr"]
    ).to(device)

    dataset_class = BatchStochasticLogLaggedPrinzDataset if config["dataset_type"] == "log" else BatchStochasticLaggedPrinzDataset
    dataset = dataset_class(path = config["data_path"], n_traj = config["n_traj"],
                            max_lag=config["max_lag"], max_length=config["traj_max_length"],
                            traj_indices = config["traj_indices"], burn_in = config["burn_in"])


    _ = pl.Trainer() #to log compute resources utiliazation

    optimizer = tddpm.configure_optimizers()
    
    best_loss = torch.inf
    for epoch in range(config["max_epochs"]):
        print(f"Epoch {epoch}")
        batches = generate_batches(dataset, config["batch_size"])
        loss_batch = torch.empty(0)
        
        for i in ( pbar := tqdm(range(len(batches))) ):
            loss = train_step_dyn(batches, i, tddpm, optimizer, device, pbar)
            loss_batch = torch.cat((loss_batch, loss.detach().cpu().unsqueeze(0)))
        mean_loss = torch.mean(loss_batch)
        if  mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(tddpm.state_dict(), model_path)
            
    wandb.log_model(name=f"model-{wandb.run.id}", path=model_path, aliases=["best"])  

    
if __name__ == "__main__":
    parser = ArgumentParser()
    #model parameters
    parser.add_argument("--model_type",      type=str, default='ito')
    parser.add_argument("--bg_id",           type=str, default='None')
    parser.add_argument("--max_eigenvalue",  type=float, default=0.994)# max eigenvalue 0.994, 0.982 for low data regime
    parser.add_argument("--diffusion_steps", type=int, default=500)
    parser.add_argument("--max_lag",         type=int, default=1000)
    #training parameters
    parser.add_argument("--emb_dim",         type=int, default=256)
    parser.add_argument("--net_dim",         type=int, default=256)
    parser.add_argument("--max_epochs",      type=int, default=1500)
    parser.add_argument("--batch_size",      type=int, default=2097152)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--data_path",       type=str, default='storage/data/prinz/trajs.npy')
    parser.add_argument("--dataset_type",    type=str, default='uniform') #log
    parser.add_argument("--n_traj", type=lambda x: int(x) if x.isdigit() else None, default=None)
    parser.add_argument('--traj_indices',    nargs='+', default=[])
    parser.add_argument("--burn_in",         type=int, default=1000)
    parser.add_argument("--traj_max_length", type=lambda x: int(x) if x.isdigit() else None, default=None)
    parser.add_argument("--no_log_epochs",   type=int, default=0)

    #other
    parser.add_argument("--tags",             nargs='+', default=[])
    parser.add_argument("--progress_bar",     action='store_true')
    parser.add_argument("--project_path",     type=str, default=os.getcwd())    
    
    args = parser.parse_args()
    args.traj_indices = [int(i) for i in args.traj_indices]
    args.tags = [str(i) for i in args.tags]
    main(args)
