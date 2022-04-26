### Code for Gromov-Wasserstein Imitation Learning, Arnaud Fickinger, 2022
# Copyright (c) Meta Platforms, Inc. and affiliates.

# !/usr/bin/env python3

import numpy as np
import torch
import os
import time
import pickle as pkl
from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils
import dmc2gym
import hydra
import wrappers


def make_maze(cfg, env_id, maze_id):
    from gym.envs.mujoco import mujoco_env
    from gym.wrappers import TimeLimit
    from maze_envs import MazeEnd_PointMass
    if env_id == 'MazeEnd_PointMass':
        env = MazeEnd_PointMass(maze_id=maze_id)
    else:
        assert False
    if cfg.time_limit > 0:
        env = TimeLimit(env, cfg.time_limit)
    return env


def make_env(cfg, env_id, maze_id=0):
    """Helper function to create dm_control environment"""
    if cfg.dmc:
        if env_id == 'ball_in_cup_catch':
            domain_name = 'ball_in_cup'
            task_name = 'catch'
        else:
            domain_name = env_id.split('_')[0]
            task_name = '_'.join(env_id.split('_')[1:])
        env = dmc2gym.make(domain_name=domain_name,
                           task_name=task_name,
                           seed=cfg.seed,
                           visualize_reward=False)
    elif 'Maze' in env_id:
        env = make_maze(cfg, env_id, maze_id)
        random_rgb = np.array([0., 0., 0., 0.])
        env.sim.model.geom_rgba[0, :] = random_rgb
    else:
        assert False
    if cfg.ultra_sparse:
        env = wrappers.SparseRewardCartpole(env)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        agent_name = cfg.agent._target_.split('.')[1]
        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=agent_name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

        # load the expert demonstration
        with open(f'{cfg.demonstration_name}', 'rb') as handle:
            dict_demonstration = pkl.load(handle)
        traj_expert = dict_demonstration['obs']

        if cfg.gw_include_actions_expert:
            traj_expert = np.concatenate((traj_expert, dict_demonstration['action']), axis=1)
        self.traj_expert = traj_expert

        self.env = make_env(cfg, cfg.env_agent, cfg.maze_id_agent)

        cfg.agent.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        if cfg.pretrained_agent != '':
            self.agent.actor.load_state_dict(
                torch.load(f'{cfg.pretrained_agent}'))

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device, cfg)

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        to_evaluate = False

        if 'Maze' in self.cfg.env_agent or self.cfg.ultra_sparse:
            episode_dense_reward = 0
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    duration = time.time() - start_time
                    self.logger.log('train/duration',
                                    duration, self.step)
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                    # evaluate agent periodically
                    if to_evaluate:
                        self.logger.log('eval/episode', episode, self.step)
                        self.evaluate()
                        to_evaluate = False

                    self.logger.log('train/episode_reward', episode_reward,
                                    self.step)

                obs = self.env.reset()

                self.agent.reset()
                done = False
                episode_reward = 0
                if 'Maze' in self.cfg.env_agent or self.cfg.ultra_sparse:
                    episode_dense_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step, gw=True,
                                  normalize_reward=self.cfg.gw_normalize,
                                  normalize_reward_batch=self.cfg.gw_normalize_batch,
                                  include_external_reward=self.cfg.include_external_reward,
                                  weight_external_reward=self.cfg.weight_external_reward,
                                  weight_gw_reward=self.cfg.weight_gw_reward)

            next_obs, reward, done, info = self.env.step(action)

            if 'Maze' in self.cfg.env_agent or self.cfg.ultra_sparse:
                episode_dense_reward += info['dense_reward']

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            if done:
                episode_gw_reward, normalized_episode_gw_reward = self.replay_buffer.process_trajectory(
                    self.traj_expert,
                    metric_expert=self.cfg.metric_expert, metric_agent=self.cfg.metric_agent,
                    include_actions=self.cfg.gw_include_actions_agent, entropic=self.cfg.gw_entropic,
                    normalize_agent_with_expert=self.cfg.normalize_agent_with_expert,
                    sinkhorn_reg=self.cfg.sinkhorn_reg)
                self.logger.log('train/episode_gw_reward', episode_gw_reward,
                                self.step)
                self.logger.log('train/normalized_episode_gw_reward', normalized_episode_gw_reward,
                                self.step)
            obs = next_obs
            episode_step += 1

            self.step += 1
            if self.cfg.eval_frequency > 0 and self.step % self.cfg.eval_frequency == 0:
                to_evaluate = True
                self.evaluate_sample = self.step


@hydra.main(config_path='config', config_name='imitate.yaml')
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
