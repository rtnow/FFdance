# -*- coding: UTF-8 -*- #
"""
@filename:FFD_train.py
@auther:Rtnow
@time:2025-02-02
"""
import warnings

from sympy.codegen import Print

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import hydra
from pathlib import Path
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import wandb
import time
import gc
from typing import List
import imageio
from collections import deque

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._obs_channel = self.train_env.observation_spec().shape[0]
        self.best_eval_reward = 0

    def setup(self):
        if self.cfg.use_wandb:  # Weights and Biases
            exp_name = '_'.join([
                self.cfg.task_name,
                str(self.cfg.seed)
            ])
            wandb.init(project="sim2real", group=self.cfg.wandb_group, name=exp_name)
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, use_wandb=self.cfg.use_wandb)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed, randomize=True, two_cam=True,
                                  img_size=self.cfg.img_size, use_depth=self.cfg.use_depth)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed, img_size=self.cfg.img_size,
                                 use_depth=self.cfg.use_depth)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None
        self.stored_episodes = deque([], maxlen=20) if self.cfg.use_traj else None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.sample_action(time_step.observation[:self._obs_channel],
                                                      time_step.observation[self._obs_channel:],
                                                      self.global_step,
                                                      eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

        if (self.best_eval_reward < (total_reward / episode)) and self.cfg.save_snapshot and self.global_step >= int(
                5e5):
            self.best_eval_reward = (total_reward / episode)
            self.save_snapshot(best=True, step=self.global_step)
            print('final period best eval reward:', self.best_eval_reward)

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        # episodic_list is used to store the observation of each episode
        episodic_list: List[np.ndarray] = []

        time_step = self.train_env.reset()
        episodic_list.append(time_step.observation[self._obs_channel:].copy())  # TODO
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                if self.cfg.use_traj:
                    self.stored_episodes.append(episodic_list)
                episodic_list = []
                # reset env
                time_step = self.train_env.reset()
                episodic_list.append(time_step.observation[self._obs_channel:].copy())  # TODO
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot and (self.global_step % int(2e4) == 0):
                    self.save_snapshot(step=self.global_step)
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):  # 1000k步eval一次
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.sample_action(time_step.observation[:self._obs_channel],
                                                  time_step.observation[-self._obs_channel:],
                                                  self.global_step,
                                                  eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):  # 4k步之后seed_until_step(self.global_step)为False，开始更新
                metrics = self.agent.update(self.replay_iter, self.global_step, traj=self.stored_episodes)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episodic_list.append(time_step.observation[self._obs_channel:].copy())  # TODO

            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self, best=False, step=None):
        if best:
            snapshot = self.work_dir / f'best_snapshot_{step}.pt'
        else:
            snapshot = self.work_dir / f'snapshot_{step}.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='FFDance_config')
def main(cfg):
    from FFD_train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()

if __name__ == '__main__':
    main()
