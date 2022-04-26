### Code for Gromov-Wasserstein Imitation Learning, Arnaud Fickinger, 2022
# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
import ot
import scipy as sp

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, cfg):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.gw_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        self.idx_gw = 0
        self.full_gw = False

        self.cfg = cfg

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        # import pdb;pdb.set_trace()
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def process_trajectory(self, traj_expert, metric_expert = 'euclidean', metric_agent = 'euclidean', sinkhorn_reg=5e-3, normalize_agent_with_expert=False, include_actions=True, entropic=True):
        assert not (self.idx == 0 and not self.full)
        if self.idx == 0:
            traj_agent = self.obses[self.idx_gw:]
        else:
            traj_agent = self.obses[self.idx_gw:self.idx]

        if normalize_agent_with_expert:
            traj_agent = (traj_agent - traj_expert.mean())/(traj_expert.std())

        if include_actions:
            if self.idx == 0:
                actions_trajectory = self.actions[self.idx_gw:]
            else:
                actions_trajectory = self.actions[self.idx_gw:self.idx]
            traj_agent = np.concatenate((traj_agent,actions_trajectory), axis=1)

        gw_rewards = self.compute_gw_reward(traj_expert, traj_agent, metric_expert, metric_agent,
                                                          entropic, sinkhorn_reg=sinkhorn_reg)

        if self.idx == 0:
            self.gw_rewards[self.idx_gw:] = np.expand_dims(gw_rewards, axis=1)
            normalized_reward = ((self.gw_rewards[:self.idx] - self.gw_rewards[:self.idx].mean())/(1e-5+self.gw_rewards[:self.idx].std()))[self.idx_gw:].sum()

        else:
            self.gw_rewards[self.idx_gw:self.idx] = np.expand_dims(gw_rewards, axis=1)
            normalized_reward = ((self.gw_rewards[:self.idx] - self.gw_rewards[:self.idx].mean())/(1e-5+self.gw_rewards[:self.idx].std()))[self.idx_gw:self.idx].sum()

        self.idx_gw = self.idx

        return gw_rewards.sum(), normalized_reward

    def compute_gw_reward(self, traj_expert, traj_agent, metric_expert = 'euclidean', metric_agent = 'euclidean', entropic=True, sinkhorn_reg=5e-3, return_coupling = False):
        distances_expert = sp.spatial.distance.cdist(traj_expert, traj_expert, metric=metric_expert)

        distances_agent = sp.spatial.distance.cdist(traj_agent, traj_agent, metric=metric_agent)

        distances_expert/=distances_expert.max()
        distances_agent/=distances_agent.max()

        distribution_expert = ot.unif(len(traj_expert))
        distribution_agent = ot.unif(len(traj_agent))

        if entropic:
            optimal_coupling = ot.gromov.entropic_gromov_wasserstein(
                distances_expert, distances_agent, distribution_expert, distribution_agent, 'square_loss', epsilon=sinkhorn_reg, max_iter=1000, tol=1e-9)
        else:
            optimal_coupling= ot.gromov.gromov_wasserstein(distances_expert, distances_agent, distribution_expert, distribution_agent, 'square_loss')


        constC, hExpert, hAgent = ot.gromov.init_matrix(distances_expert, distances_agent, distribution_expert, distribution_agent, loss_fun='square_loss')

        tens = ot.gromov.tensor_product(constC, hExpert, hAgent, optimal_coupling)

        rewards = -(tens*optimal_coupling).sum(axis=0)

        if return_coupling:
            return rewards, optimal_coupling

        return rewards

    def sample(self, batch_size, gw=False, normalize_reward=False,normalize_reward_batch=False, include_external_reward=False, weight_external_reward=1, weight_gw_reward=1):

        if gw:
            end_idxs = self.capacity if self.full_gw else self.idx_gw
        else:
            end_idxs = self.capacity if self.full else self.idx

        idxs = np.random.randint(0,
                                 end_idxs,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        if gw:
            if normalize_reward_batch:
                rewards = torch.as_tensor((self.gw_rewards[idxs] - self.gw_rewards[idxs].mean())/(1e-5+self.gw_rewards[idxs].std()), device=self.device)
            elif normalize_reward:
                gw_rewards_normalized = (self.gw_rewards[:end_idxs] - self.gw_rewards[:end_idxs].mean())/(1e-5+self.gw_rewards[:end_idxs].std())
                rewards = torch.as_tensor(gw_rewards_normalized[idxs], device=self.device)
            else:
                rewards = torch.as_tensor(self.gw_rewards[idxs], device=self.device)

        else:
            rewards = torch.as_tensor(self.rewards[idxs], device=self.device)

        if include_external_reward:
            assert gw
            rewards=weight_gw_reward*rewards+weight_external_reward*torch.as_tensor(self.rewards[idxs], device=self.device)

        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
