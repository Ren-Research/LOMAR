# Tutorial 

Here we provide a brief tutorial about our [codebase](https://github.com/lyeskhalil/CORL). If you are already familiar with the codes and the framework, please skip this tutorial.

## 1. Bipartite Data Generation Code


### Supported Args
```
problem: "adwords", "e-obm", "osbm"
graph_family: "er", "ba", "triangular", "thick-z", "movielense", "gmission"
weight_distribution: "normal", "power", "degree", "node-normal", "fixed-normal"
weight_distribution_param: ...
graph_family_parameter: ...
```
Caution: We only list all the feasible parameters here, some combination may not be allowed (adwords + gmission), please check before using these arguments.

## 2. Networkx Guide
### Print information for a graph
```python
print(G)               
#Graph named 'triangular_graph(3,9,-1.0)' with 12 nodes and 18 edges
print(G.graph)         
#{'name': 'triangular_graph(3,9,-1.0)'}
print(G.nodes)         
#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
print(G.nodes[1])      #{'bipartite': 0}
print(G.edges)         
#[(0, 3), (0, 4), (0, 5), (1, 3), ..., (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11)]
print(G.edges[0,3])    
#{'weight': 0.3958830486584797}
```


### Convert data type
Convert the networkx graph to torch
```python
def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance"""
    pass
```
Reveal useful information from the `TORCH_GEOMETRIC.DATA` class
```python
print(data.edge_index)   # Print the edge between nodes, the graph is directed (two edges needed for one pair of nodes)
print(data.weight)       # Print the weight 
print(data.bipartite)    # Print the portion
print(data.x)            # Capacity of each resource
print(data.y)            # Optimal solution, (total_gain, action_array)
print(data.edge_attr)    # None for this project
```

## 3. Important Arguments for the OBM codebase
```shell
'load_path': 'path_to_pretrained_model',  # Model path
'checkpoint_epochs': 0,                   # The frequency of routine checkpoint  saving
'model': 'inv-ff-hist',                   # Model type
```


## 4. RL Model For OBM 
### Apply for a new GPU node in SLURM cluster (skip this step on your own server)
``` shell
srun -p gpu --gres=gpu:1 --mem=100g --nodelist=gpu02 --time=48:00:00 --pty bash -l
```

### Quick start - Data Generation 
```shell
python data/generate_data.py --problem adwords --dataset_size 1000 \
    --dataset_folder dataset/train/adwords_triangular_uniform_0.10.4_10by100/parameter_-1 \
    --u_size 10 --v_size 60 --graph_family triangular --weight_distribution uniform \
    --weight_distribution_param 0.1 0.4 --graph_family_parameter -1 

python data/generate_data.py --problem adwords --dataset_size 1 \
     --dataset_folder dataset/train/adwords_triangular_triangular_0.10.4_10by100/parameter_-1 \
     --u_size 3 --v_size 9 --graph_family ba --weight_distribution uniform \
     --weight_distribution_param 0 1 --graph_family_parameter 2
```

### Quick Start - Training 
```shell
python run.py --encoder mpnn --model inv-ff-hist --problem adwords --batch_size 100 --embedding_dim 30 --n_heads 1 --u_size 10  --v_size 60 \
            --n_epochs 20 --train_dataset dataset/train/adwords_triangular_uniform_0.10.4_10by60/parameter_-1 \
            --val_dataset dataset/val/adwords_triangular_uniform_0.10.4_10by60/parameter_-1 \
            --dataset_size 1000 --val_size 100 --checkpoint_epochs 0 --baseline exponential --lr_model 0.006 --lr_decay 0.97 \
            --output_dir saved_models --log_dir logs_dataset --n_encode_layers 1 \
            --save_dir saved_models/adwords_triangular_uniform_0.10.4_10by60/parameter_-1 \
            --graph_family_parameter -1 --exp_beta 0.8 --ent_rate 0.0006
```

### Quick start - Evaluation
* Note: Please first train the model, if you already have the model, you can use the following code to validate the performance. Don't forget to replace the load_path and other arguments accordingly.
```shell
python3 run_lite.py --encoder mpnn --model inv-ff-hist-switch --problem osbm --batch_size 100 --embedding_dim 30 --n_heads 1 --u_size 10  --v_size 60 --n_epochs 300 --train_dataset dataset/train/osbm_movielense_default_-1_10by60/parameter_-1 --val_dataset dataset/val/osbm_movielense_default_-1_10by60/parameter_-1 --dataset_size 20000 --val_size 1000 --checkpoint_epochs 0 --baseline exponential  --lr_model 0.006 --lr_decay 0.97 --output_dir saved_models --log_dir logs_dataset --n_encode_layers 1 --save_dir saved_models/osbm_movielense_default_-1_10by60/parameter_-1 --graph_family_parameter -1 --exp_beta 0.8 --ent_rate 0.0006 --eval_only --no_tensorboard --load_path saved_models/inv-ff-hist/run_20220501T195416/best-model.pt --switch_lambda 0.0 --slackness 0.0 --max_reward 5.0
```



