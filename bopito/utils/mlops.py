import hashlib
import os
import pickle as pkl
import re
import shutil
from datetime import datetime

import pytorch_lightning as pl

import wandb

def hash_obj(dict_):
    d_str = pkl.dumps(dict_)
    return hash(d_str)

def init_wandb(project, config=None, wandb_off=False, group=None):
    mode = "disabled" if wandb_off else None
    if config is not None:
        if "tags" in config:
            run = wandb.init(project=project, mode=mode, group=group, config=config, tags=config["tags"])
        else:
            run = wandb.init(project=project, mode=mode, group=group, config=config)
    else:
        run = wandb.init(project=project, mode=mode, group=group)
    wandb.run.name = re.sub(r"\d+$", wandb.run.id, wandb.run.name)
    print(f"wandb run name: {wandb.run.name}")
    print(f"wandb run id:   {wandb.run.id}")
    print(f"pid:            {os.getpid()}")
    return run


def get_wandb_logger(project, wandb_off=False, group=None, config=None):
    run= init_wandb(project=project, wandb_off=wandb_off, group=group, config=config)
    logger = pl.loggers.WandbLogger(log_model="all")
    return logger, run


def get_checkpoint(project, run_id, tag="best"):
    artifact_dir = get_artifact(project, run_id, tag=tag).download()
    artifact_dir = fix_artifact_dir(artifact_dir)
    return os.path.join(artifact_dir, "model.ckpt")

def get_model(run_id, tag, model_cls):
    ckpt = get_checkpoint("test", run_id, tag=tag)
    model = model_cls.load_from_checkpoint(checkpoint_path=ckpt)
    print("Model loaded ...")
    return model

def get_checkpoint_config(project, run_id, tag="best"):
    artifact_dir = get_artifact(project, run_id, tag).download()
    return os.path.join(artifact_dir, "model.pth")

def get_artifact(project, run_id, tag="best"):
    api = wandb.Api()
    artifact = api.artifact(
        os.path.join(project, f"model-{run_id}:{tag}"), type="model"
    )
    return artifact

##############################
# there is currently a bug in pl such that checkpoints with a ":" in the name cannot
# be loaded into a model. This is a sloppy workaround.
def fix_artifact_dir(artifact_dir):
    new_path = artifact_dir.replace(":", "_")
    if not os.path.exists(new_path):
        shutil.move(artifact_dir, new_path)
    artifact_dir = new_path
    return artifact_dir


##############################


#def get_artifact(project, run_id, name, tag="best"):
#    api = wandb.Api()  # pylint: disable=c-extension-no-member
#    artifact = api.artifact(
#        os.path.join(project, f"{name}-{run_id}:{tag}"), type="model"
#    )
#    return artifact


def get_timestamp():
    return datetime.now().strftime("%y%m%d.%H%M%S")


class ResultsHandler:
    def __init__(self, job, experiment=None, timestamp=True):
        experiment = "" if experiment is None else experiment
        timestamp = get_timestamp() if timestamp else ""
        self.path = os.path.join("experiments", job+"/",experiment +"_"+timestamp)
        os.makedirs(self.path, exist_ok=True)

        print("results path in", self.path)

    def get_results_path(self):
        return self.path

    def get_path(self, name):
        return os.path.join(self.path, name)

    def save(self, obj, name):
        path = self.get_path(name)
        save(obj, path)

    def load(self, name):
        path = self.get_path(name)
        return load(path)

    def slurm(self):
        if os.environ.get("SLURM_DIR") is not None:
            slurm_dir = os.environ.get("SLURM_DIR")
            shutil.move(slurm_dir, os.path.join(self.path, "slurm"))


def get_results_path(job, timestamp=True, experiment=None):
    experiment = "" if experiment is None else experiment
    timestamp = get_timestamp()
    path = os.path.join("experiments", experiment, job, timestamp)

    os.makedirs(path, exist_ok=True)

    print("results path in", path)
    return path


def move_slurm_output(result_dir):
    if os.environ.get("SLURM_DIR") is not None:
        slurm_dir = os.environ.get("SLURM_DIR")
        shutil.move(slurm_dir, os.path.join(result_dir, "slurm"))


def hash_model(model):
    hasher = hashlib.sha256()
    for param in model.parameters():
        hasher.update(param.data.cpu().numpy())

    return hasher.hexdigest()[:8]


def save(obj, name=None):
    if name is None:
        name = "tmp.pkl"
    name = "./" + name
    os.makedirs(os.path.dirname(name), exist_ok=True)
    with open(f"{name}", "wb") as f:
        pkl.dump(obj, f)


def load(name=None):
    if name is None:
        name = "tmp.pkl"
    with open(f"{name}", "rb") as f:
        return pkl.load(f)
