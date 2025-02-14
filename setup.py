from setuptools import find_packages, setup

setup(
    name="bopito",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pytorch_lightning",
        "torch_geometric",
        "mdshare",
        "mdtraj",
        "tqdm",
        "deeptime",
        "matplotlib",
        "h5py",
        "wandb",
        "ase",
        "rdkit",
        "lovely_tensors",
        "e3nn",
        "POT",
    ],
)
