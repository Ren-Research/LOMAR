import torch
from torch import nn
from policy.inv_ff_history import InvariantFFHist
from utils.functions import random_max
import math

# from utils.visualize import visualize_reward_process
from copy import deepcopy

class InvariantFFHistSwitch(InvariantFFHist):
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
        normalization="batch",
        checkpoint_encoder=False,
        shrink_size=None,
        num_actions=4,
        n_heads=None,
        encoder=None,
    ):
        super(InvariantFFHistSwitch, self).__init__(embedding_dim,
            hidden_dim,
            problem,
            opts,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            n_encode_layers=n_encode_layers,
            normalization=normalization,
            checkpoint_encoder=checkpoint_encoder,
            shrink_size=shrink_size,
            num_actions=num_actions,
            n_heads=n_heads,
            encoder=encoder)
        
        self.switch_lambda = opts.switch_lambda
        self.slackness     = opts.slackness
        self.max_reward    = opts.max_reward
        self.softmax_temp  = opts.softmax_temp
        

    def _inner_pure_ml(self, input, opts, show_rewards = False):

        outputs = []
        sequences = []
        state = self.problem.make_state(deepcopy(input), opts.u_size, opts.v_size, opts)

        # Perform decoding steps
        rewards = []
        i = 1
        while not (state.all_finished()):
            mask = state.get_mask()
            state.get_current_weights(mask)
            s, mask = state.get_curr_state(self.model_name)
            pi = self.ff(s).reshape(state.batch_size, state.u_size + 1)
            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected, p = self._select_node(
                pi, mask.bool()
            )  # Squeeze out steps dimension
            # entropy += torch.sum(p * (p.log()), dim=1)
            state = state.update((selected)[:, None])
            outputs.append(p)
            sequences.append(selected)
            i += 1

            rewards.append(state.size.view(-1))
        
        res_list = [torch.stack(outputs, 1),
                    torch.stack(sequences, 1),
                    state.size,]
        if show_rewards:
            # visualize_reward_process(torch.stack(rewards, 1).cpu().numpy(), "temp_ml_movie.png",ylim = [-1, 14]) 
            res_list.append(torch.stack(rewards, 1))
        
        # Collected lists, return Tensor
        return res_list
    
    def _inner_switch(self, input, opts, show_rewards = False, osm_baseline = False):
        
        # torch.autograd.set_detect_anomaly(True)

        if opts.problem not in ["osbm", "e-obm"]:
            raise NotImplementedError
            
        batch_size = int(input.batch.size(0) / (opts.u_size + opts.v_size + 1) )

        ## Greedy algorithm state
        state_greedy = self.problem.make_state(deepcopy(input), opts.u_size, opts.v_size, opts)
        
        ## ML algorithm state
        state_ml = self.problem.make_state(deepcopy(input), opts.u_size, opts.v_size, opts)
        sequences_ml, outputs_ml, rewards_ml = [], [], []

        graph_v_size = opts.v_size
        i = 1

        while not (state_ml.all_finished()):
            #Extract Current Info
            mask_gd = state_greedy.get_mask()
            weight_gd = state_greedy.get_current_weights(mask_gd).clone()
            weight_gd[mask_gd.bool()] = -1.0

            if osm_baseline:
                # Select action based on current info
                if(i > graph_v_size/math.e):
                    selected_gd = random_max(weight_gd)
                else:
                    selected_gd = torch.zeros([weight_gd.shape[0], 1], device=weight_gd.device, dtype=torch.int64)
            else:
                selected_gd = random_max(weight_gd)
            
            state_greedy = state_greedy.update(selected_gd)

            # Generate Probabality from Greedy algorithm
            probs_gd = -1e8 * torch.ones(mask_gd.shape, device = mask_gd.device)
            probs_gd[(torch.arange(batch_size) , selected_gd.view(-1) )] = 1
            p_gd = torch.log_softmax(probs_gd, dim=1)
            
            reward_gd = state_greedy.size

            #Extract Current Info
            mask_ml = state_ml.get_mask()
            state_ml.get_current_weights(mask_ml)
            info_s, mask_ml = state_ml.get_curr_state(self.model_name)
            pi = self.ff(info_s).reshape(state_ml.batch_size, state_ml.u_size + 1)
            
            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected_ml, p_ml = self._select_node(pi, mask_ml.bool())  
            selected_ml = (selected_ml)[:, None]

            # Collect reward based on these hypothesis actions
            reward_hyp, matched_node_hyp = state_ml.try_hypothesis_action(selected_ml)

            # Compare the ML with Greedy actions
            matched_node_gd = state_greedy.matched_nodes
            action_diff = torch.relu(matched_node_hyp - matched_node_gd)[:,1:]
            action_diff = action_diff.sum(1, keepdim=True)

            # Calculate reward
            reward_gd_reserve = reward_gd + action_diff * self.max_reward
            reward_diff = reward_hyp - (self.switch_lambda * reward_gd_reserve - self.slackness)
            
            # Determin the probability of the final action
            probs_policy       = torch.zeros([batch_size, 2], device=mask_gd.device)
            probs_policy[:,0]  = reward_diff.view(-1)/self.softmax_temp  # Normalize the selection cost
            p_policy_exp       = torch.log_softmax(probs_policy, dim=1).exp()
            
            p_total_exp        = p_policy_exp[:,0].view(-1,1) * (p_ml.exp()) + p_policy_exp[:,1].view(-1,1) * (p_gd.exp()) + 1e-8
            p_total            = p_total_exp.log()

            # Determine the action based on the reward_diff (expert v.s. ML)
            if (self.ff.training):
                # During training, use probability based method
                projected_ml, p_final = self._select_node(p_total, mask_ml.bool())
                projected_ml = (projected_ml)[:, None]
            else:
                # During training, use hard-switch
                reward_sign    = (reward_diff >= 0)
                avaliable_sign = (matched_node_hyp.gather(1, selected_gd) < 1e-10)
                expert_sign    = torch.logical_not(reward_sign) * avaliable_sign
                projected_ml   = selected_ml*reward_sign + selected_gd * expert_sign 
                p_final        = torch.log_softmax(p_total, dim=1)
            
            # Update the current state
            state_ml = state_ml.update(projected_ml)       
            outputs_ml.append(p_final)
            sequences_ml.append(projected_ml.squeeze(1))  
            i += 1

            rewards_ml.append(state_ml.size.view(-1))

        res_list = [torch.stack(outputs_ml, 1),
                    torch.stack(sequences_ml, 1),
                    state_ml.size,]

        if show_rewards:
            # visualize_reward_process(torch.stack(rewards, 1).cpu().numpy(), "temp_ml_movie.png",ylim = [-1, 14]) 
            res_list.append(torch.stack(rewards_gd, 1))
        
        # Collected lists, return Tensor
        return res_list

    def _inner_greedy(self, input, opts, show_rewards = False):
        state = self.problem.make_state(deepcopy(input), opts.u_size, opts.v_size, opts)
        sequences = []
        outputs = []
        batch_size = int(input.batch.size(0) / (opts.u_size + opts.v_size + 1) )

        rewards = []
        i = 1
        while not (state.all_finished()):
            mask = state.get_mask()
            w = state.get_current_weights(mask).clone()
            w[mask.bool()] = -1.0
            selected = random_max(w)
            state = state.update(selected)
            sequences.append(selected.squeeze(1))

            # Generate Probabality from Greedy algorithm
            probs = -1e8 * torch.ones(mask.shape, device = mask.device)
            probs[(torch.arange(batch_size) , selected.view(-1) )] = 1
            p = torch.log_softmax(probs, dim=1)
            outputs.append(p)
            i += 1

            rewards.append(state.size.view(-1))
        
        res_list = [torch.stack(outputs, 1),
                    torch.stack(sequences, 1),
                    state.size,]

        if show_rewards:
            # visualize_reward_process(torch.stack(rewards, 1).cpu().numpy(), "temp_ml_movie.png",ylim = [-1, 14]) 
            res_list.append(torch.stack(rewards, 1))
        
        # Collected lists, return Tensor
        return res_list

    def _inner(self, input, opts):

        # visualize_reward_process(torch.stack(rewards, 1).cpu().numpy(), "temp_ml_movie.png",ylim = [-1, 14]) 

        # return self._inner_greedy(input, opts)
        # return self._inner_pure_ml(input, opts)
        return self._inner_switch(input, opts)
 
