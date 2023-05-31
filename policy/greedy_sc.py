import torch
from torch import nn
from utils.functions import random_max
import math 

class GreedySC(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        opts,
        tanh_clipping=None,
        mask_inner=None,
        mask_logits=None,
        n_encode_layers=None,
        normalization=None,
        checkpoint_encoder=False,
        shrink_size=None,
        num_actions=None,
        n_heads=None,
        encoder=None,
    ):
        super(GreedySC, self).__init__()
        self.decode_type = None
        self.problem = problem
        self.model_name = "greedy-sc"

    def forward(self, x, opts, optimizer, baseline, return_pi=False):
        state = self.problem.make_state(x, opts.u_size, opts.v_size, opts)
        sequences = []

        # Array for single step rewards
        i = 0
        reward_array = torch.zeros([opts.v_size, opts.batch_size])
        prev_reward = 0

        graph_v_size = opts.v_size
        i = 0

        while not (state.all_finished()):
            mask = state.get_mask()
            w = state.get_current_weights(mask).clone()
            w[mask.bool()] = -1.0

            # Select action based on current info
            if(i > graph_v_size/math.e - 1):
                selected = random_max(w)
            else:
                selected = torch.zeros([w.shape[0], 1], device=w.device, dtype=torch.int64)

            state = state.update(selected)
            sequences.append(selected.squeeze(1))

            # Collect the signle step reward
            reward_array[i,:] = state.size.view(-1).cpu() - prev_reward
            prev_reward = state.size.view(-1).cpu()
            i+= 1

        if return_pi:
            return -state.size, None, torch.stack(sequences, 1), None

        return -state.size, torch.stack(sequences, 1), None

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
