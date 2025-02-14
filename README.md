## Prerequisites
* Python 3.11.
* CUDA-enabled GPU.

## Installation
Follow these steps to set up the environment and install the necessary dependencies for BoPITO.

### Installing Dependencies
To install the required dependencies, run:

```
$ cd bopito
$ make install
```

This command will install all package dependencies. If you find issues installing pytorch-scatter, pytorch-sparse, pytorch, and pytorch-cluster, which are architecture-specific dependencies for pytorch-geometric, visit [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).


## Usage
The [./scripts][(./scripts)] directory contains a sub-directory for the Prinz Potential ([prinz](./scripts/prinz)) and one for Alanine Dipeptide ([ala2](./scripts/ala2)). Both subdirectories contain two scripts, train.py and sample.py. Evaluation functions are provided in [bopito/evaluation/](./bopito/evaluation/).

### Train a model
To train a model use the following command:

```
python scripts/{system}/train.py --model_type {model_type}
```
{system} is prinz or ala2 and {model_type} is ito, bg or bopito. You need to train a bg model before you can train a bopito model. To train a bopito model you need to add the argument --bg_id {bg_id} to previous command, where {bg_id} is the wandb run of your bg model.

These scripts come with several arguments to customize training. To see all available arguments run:

```
python scripts/{system}/train.py --help
```

### Sampling from model
To sample a model, use the following command:

```
python scripts/{system}/sample.py {run_id} --model_type {model_type}
```
where {run_id} is the wanbd id from your model.

For sampling options please check 
```
python scripts/{system}/sample.py --help
```

