import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from train import rollout
import torch 

def visualize_reward_process(reward_array, fig_path, ylim = None):
    fig, ax = plt.subplots(1)
    x_array = np.arange(reward_array.shape[1])
    for k in range(reward_array.shape[0]):
        ax.plot(x_array, reward_array[k, :], label = "sample_{}".format(k))
    
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.legend()
    fig.show()
    plt.savefig(fig_path)
    plt.close()

def validate_histogram(models, model_names, dataset, opts):
    assert len(models) == len(model_names)
    num_models = len(models)

    cost_list = []
    cr_list   = []
    for i in range(num_models):
        model       = models[i]
        model_name  = model_names[i]
        cost, cr, _ = rollout(model, dataset, opts)
        print("{} Validation overall avg_cost: {} +- {}".format(model_name, 
                cost.mean(), torch.std(cost)))

        cost_list.append(cost)
        cr_list.append(cr)

    bins = 30
    fig, ax = plt.subplots(1)
    for i in range(num_models):
        model_name  = model_names[i]
        cost        = cost_list[i]
        cost        = -cost.view(-1).cpu().numpy()
        ax.hist(cost, bins = bins, density=True, alpha = 0.4, label = model_name)

    plt.legend()
    fig.show()
    plt.savefig("temp_hist_cost.png")
    plt.close()

