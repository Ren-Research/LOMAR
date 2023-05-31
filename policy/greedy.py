import torch
from torch import nn
from utils.functions import random_max


class Greedy(nn.Module):
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
        super(Greedy, self).__init__()
        self.decode_type = None
        self.problem = problem
        self.model_name = "greedy"

    def forward(self, x, opts, optimizer, baseline, return_pi=False):
        state = self.problem.make_state(x, opts.u_size, opts.v_size, opts)
        sequences = []

        # Array for single step rewards
        i = 0
        reward_array = torch.zeros([opts.v_size, opts.batch_size])
        prev_reward = 0

        while not (state.all_finished()):
            mask = state.get_mask()
            w = state.get_current_weights(mask).clone()
            w[mask.bool()] = -1.0
            selected = random_max(w)
            state = state.update(selected)
            sequences.append(selected.squeeze(1))

            # Collect the signle step reward
            reward_array[i,:] = state.size.view(-1).cpu() - prev_reward
            prev_reward = state.size.view(-1).cpu()
            i+= 1

        if return_pi:
            return -state.size, None, torch.stack(sequences, 1), None

        # import os
        # base_dir = "reward_save"
        # file_list = os.listdir(base_dir)
        # if file_list.__len__() != 0:
        #     file_list.sort()
        #     curr_index = int(file_list[-1][-6:-3]) + 1
        # else:
        #     curr_index = 0
        # torch.save(reward_array, base_dir+"/reward_{:0>3d}.pt".format(curr_index))

        return -state.size, torch.stack(sequences, 1), None

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
