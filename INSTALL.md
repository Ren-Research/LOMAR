# Installation Guide

## Install Online Bipartite Matching code base
```bash
    
    conda create --name renlab_obm python==3.7.0
    conda activate renlab_obm
    # Install pytorch
    pip install torch==1.11.0 torchvision
    # Install Pytorch Geometric
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
    # CPU-only version
    # pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html

    # Install gurobi
    pip install gurobipy
    # Download seperate license verification tool.
    wget https://packages.gurobi.com/lictools/licensetools9.5.1_linux64.tar.gz
    # Please obtain a gurobi key before proceeding, replace the `your_gurobi_key` with your key
    grbgetkey your_gurobi_key
    
    # Install other requirements
    pip install -r requirements.txt

    pip install wandb
    
    pip install -v -e .
```

## Request an GPU node in Slurm Cluster

```shell
srun -p gpu --gres=gpu:1 --mem=100g --time=48:00:00 --pty bash -l
```


