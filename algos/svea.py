# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algos.color_jitter import random_color_jitter
from algos.encoder import make_encoder

import utils
from utils import random_overlay, random_mask_freq_v2

def _get_out_shape(in_shape, layers, attn=False):
    x = torch.randn(*in_shape).unsqueeze(0)
    if attn:
        return layers(x, x, x).squeeze(0).shape
    else:
        return layers(x).squeeze(0).shape

class NormalizeImg(nn.Module):
    def __init__(self, mean_zero=False):
        super().__init__()
        self.mean_zero = mean_zero

    def forward(self, x):
        if self.mean_zero:
            return x/255. - 0.5
        return x/255.

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Actor(nn.Module):
    def __init__(self, repr_dim, feature_dim, hidden_dim, action_shape):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0])
        )

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.layers(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        # 检查批次数是否相等
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    def __init__(self, repr_dim, feature_dim, hidden_dim, action_shape):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = QFunction(
            feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            feature_dim, action_shape[0], hidden_dim
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        q1 = self.Q1(h, action)
        q2 = self.Q2(h, action)

        return q1, q2


class SVEAAgent:
    def __init__(self,
                 device, critic_target_tau, encoder_tau,
                 update_every_steps, use_tb, use_wandb, num_expl_steps,
                 stddev_schedule, stddev_clip,
                 obs_shape, action_shape, hidden_dim, projection_dim,
                 num_shared_layers, num_head_layers, num_filters,
                 context_third, context_ego, encoder_type,
                 init_temperature, multiview,
                 actor_lr, critic_lr, alpha_lr, encoder_lr,
                 aux_coef, aux_l2_coef, aux_latency):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        # models
        self.encoder = make_encoder(
            encoder_type,
            obs_shape,
            num_shared_layers,
            num_head_layers,
            num_filters,
            context_third,
            context_ego,
            mean_zero=False,
            attention=True,
            multiview=multiview,
            cat=True,
            use_resnet=True
        ).to(device)  # TODO
        self.actor = Actor(self.encoder.out_dim, projection_dim, hidden_dim, action_shape).to(device)
        self.critic = Critic(self.encoder.out_dim, projection_dim, hidden_dim, action_shape).to(device)
        self.critic_target = Critic(self.encoder.out_dim, projection_dim, hidden_dim, action_shape).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def sample_action(self, fix_obs, ego_obs, step, eval_mode):
        fix_obs = torch.as_tensor(fix_obs, device=self.device)
        ego_obs = torch.as_tensor(ego_obs, device=self.device)
        fix_obs = fix_obs.unsqueeze(0)
        ego_obs = ego_obs.unsqueeze(0)
        obs = self.encoder(fix_obs, ego_obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step, aug_obs):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        # obs = utils.cat(obs, aug_obs)
        # action = utils.cat(action, action)
        # target_Q = utils.cat(target_Q, target_Q)
        # Q1, Q2 = self.critic(obs, action)
        # critic_loss = (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q))

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        aug_Q1, aug_Q2 = self.critic(aug_obs, action)
        aug_loss = F.mse_loss(aug_Q1, target_Q) + F.mse_loss(aug_Q2, target_Q)

        critic_loss = 0.5 * (critic_loss + aug_loss)


        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step, traj):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        l = obs.shape[1] // 3
        ego = None
        fix_obs = obs.float()[:, :l]
        aug_obs = self.aug(fix_obs.float())
        original_obs = aug_obs.clone()
        fix_next_obs = next_obs.float()[:, :l]
        aug_next_obs = self.aug(fix_next_obs.float())

        # strong augmentation
        if l % 3 == 0:
            strong_aug_obs = self.encoder(random_overlay(original_obs), ego_obs=None)
        else:
            strong_aug_obs = random_overlay(original_obs[:, :l - 1])
            strong_aug_obs = torch.cat([strong_aug_obs, original_obs[:, l - 1:l]], dim=1)
            strong_aug_obs = self.encoder(strong_aug_obs, ego_obs=None)

        # encode
        aug_obs = self.encoder(aug_obs, ego_obs=None)
        with torch.no_grad():
            aug_next_obs = self.encoder(aug_next_obs, ego_obs=None)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(aug_obs, action, reward, discount, aug_next_obs, step, strong_aug_obs))

        # update actor
        metrics.update(self.update_actor(aug_obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
