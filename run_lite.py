import os
import numpy as np
import json
import pprint as pp

import torch
import torch.optim as optim
from itertools import product
import wandb

# from tensorboard_logger import Logger as TbLogger
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader as geoDataloader

# from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from utils.reinforce_baselines import (
    NoBaseline,
    ExponentialBaseline,
    RolloutBaseline,
    WarmupBaseline,
    GreedyBaseline,
)

from policy.attention_model import AttentionModel as AttentionModelgeo
from policy.ff_model import FeedForwardModel
from policy.ff_model_invariant import InvariantFF
from policy.ff_model_hist import FeedForwardModelHist
from policy.inv_ff_history import InvariantFFHist
from policy.inv_ff_history_switch import InvariantFFHistSwitch
from policy.greedy import Greedy
from policy.greedy_rt import GreedyRt
from policy.greedy_sc import GreedySC
from policy.greedy_theshold import GreedyThresh
from policy.greedy_matching import GreedyMatching
from policy.simple_greedy import SimpleGreedy
from policy.supervised import SupervisedModel
from policy.ff_supervised import SupervisedFFModel
from policy.gnn_hist import GNNHist
from policy.gnn_simp_hist import GNNSimpHist
from policy.gnn import GNN

# from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils.functions import torch_load_cpu, load_problem
from utils.visualize import validate_histogram


def run(opts):

    # Pretty print the run args
    # pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    # torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)
    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = SummaryWriter(
            os.path.join(
                opts.log_dir,
                opts.model,
                opts.run_name,
            )
        )
    if not opts.eval_only and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
        # Save arguments so exact configuration can always be found
        with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
            json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert (
        opts.load_path is None or opts.resume is None
    ), "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print("  [*] Loading data from {}".format(load_path))
        load_data = torch_load_cpu(load_path)
    
    assert (opts.tune_wandb == False 
            and opts.tune == False
            and opts.tune_baseline == False), "Unsupported Mode, please use \"run.py\" instead of \"run_lite.py\" "

    # assert(opts.model == "inv-ff-hist"), "We only support inv-ff-hist model up to now..."
    # Initialize model
    model_class = {
        "attention": AttentionModelgeo,
        "ff": FeedForwardModel,
        "greedy": Greedy,
        "greedy-rt": GreedyRt,
        "greedy-sc": GreedySC,
        "greedy-t": GreedyThresh,
        "greedy-m": GreedyMatching,
        "simple-greedy": SimpleGreedy,
        "inv-ff": InvariantFF,
        "inv-ff-hist": InvariantFFHist,
        "ff-hist": FeedForwardModelHist,
        "supervised": SupervisedModel,
        "ff-supervised": SupervisedFFModel,
        "gnn-hist": GNNHist,
        "gnn-simp-hist": GNNSimpHist,
        "gnn": GNN,
        "inv-ff-hist-switch":InvariantFFHistSwitch
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    # if not opts.tune:
    model, lr_schedulers, optimizers, val_dataloader, baseline = setup_training_env(
        opts, model_class, problem, load_data, tb_logger
    )

    training_dataset = problem.make_dataset(
        opts.train_dataset, opts.dataset_size, opts.problem, seed=None, opts=opts
    )
    if opts.eval_only:

        print("Evaluate Greedy algorithm as Baseline algorithm")
        greedy_model = Greedy(opts.embedding_dim, opts.hidden_dim, problem=problem, opts=opts).to(opts.device)  
        validate(greedy_model, val_dataloader, opts)

        print("Evaluate ML model from checkpoint")
        validate(model, val_dataloader, opts)
        switch_lambda_array = np.arange(0.0, 1.01, 0.1)

        


    else:
        best_avg_cr = 0.0
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            training_dataloader = geoDataloader(
                baseline.wrap_dataset(training_dataset),
                batch_size=opts.batch_size,
                num_workers=0,
                shuffle=True,
            )
            avg_reward, min_cr, avg_cr, loss = train_epoch(
                model,
                optimizers,
                baseline,
                lr_schedulers,
                epoch,
                val_dataloader,
                training_dataloader,
                problem,
                tb_logger,
                opts,
                best_avg_cr,
            )
            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            best_avg_cr = max(best_avg_cr, avg_cr)


def setup_training_env(opts, model_class, problem, load_data, tb_logger):
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem=problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        num_actions=opts.u_size + 1,
        n_heads=opts.n_heads,
        encoder=opts.encoder,
        opts=opts,
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get("model", {})})

    # Initialize baseline
    if opts.baseline == "exponential":
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == "greedy":
        baseline_class = {"e-obm": Greedy, "obm": SimpleGreedy}.get(opts.problem, None)

        greedybaseline = baseline_class(
            opts.embedding_dim,
            opts.hidden_dim,
            problem=problem,
            n_encode_layers=opts.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=opts.normalization,
            tanh_clipping=opts.tanh_clipping,
            checkpoint_encoder=opts.checkpoint_encoder,
            shrink_size=opts.shrink_size,
            num_actions=opts.u_size + 1,
            # n_heads=opts.n_heads,
        )
        baseline = GreedyBaseline(greedybaseline, opts)
    elif opts.baseline == "rollout":
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(
            baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta
        )

    # Load baseline from data, make sure script is called with same type of baseline
    if "baseline" in load_data:
        baseline.load_state_dict(load_data["baseline"])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{"params": model.parameters(), "lr": opts.lr_model}]
        + (
            [{"params": baseline.get_learnable_parameters(), "lr": opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if "optimizer" in load_data:
        optimizer.load_state_dict(load_data["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: opts.lr_decay ** epoch
    )

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        opts.val_dataset, opts.val_size, opts.problem, seed=None, opts=opts
    )
    val_dataloader = geoDataloader(
        val_dataset, batch_size=opts.batch_size, num_workers=1
    )
    assert (opts.resume is None), "Resume not supported, please use \"run.py\" instead of \"run_lite.py\"" 

    return (
        model,
        [lr_scheduler],
        [optimizer],
        val_dataloader,
        baseline,
    )


if __name__ == "__main__":
    run(get_options())
