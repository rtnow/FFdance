# -*- coding: UTF-8 -*- #
"""
@filename:FFdance.py
@auther:Rtnow
@time:2025-01-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils
from utils import random_overlay, random_mask_freq_v2
from algos.encoder import make_encoder


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


class FFD(nn.Module):
    """
    FFDance
    """
    def __init__(
            self, z_dim, encoder
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_target = encoder
        # 这里的z_dim是encoder输出的特征长度，用于对比学习计算
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))

    def encode(self, obs, ego_obs, detach=False, ema=False):
        """
        # Encoder: z_t = e(x_t, y_t)
        # :param obs, ego_obs: x_t, y_t, x y coordinates
        # :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(obs, ego_obs)
        else:
            z_out = self.encoder(obs, ego_obs)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class FFDAgent:
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
        self.encoder_tau = encoder_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb or use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.aux_coef = aux_coef
        self.aux_l2_coef = aux_l2_coef
        self.aux_latency = aux_latency

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
            cat=True
        ).to(device)  # TODO
        self.actor = Actor(self.encoder.out_dim, projection_dim, hidden_dim, action_shape).to(device)
        self.critic = Critic(self.encoder.out_dim, projection_dim, hidden_dim, action_shape).to(device)
        self.critic_target = Critic(self.encoder.out_dim, projection_dim, hidden_dim, action_shape).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # 处理温度系数
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr
        )
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=encoder_lr
        )

        # create FFD encoder
        self.FFD = FFD(
            self.encoder.out_dim, self.encoder
        ).to(self.device)
        self.FFD_optimizer = torch.optim.Adam(
            self.encoder.third_cnn.parameters(),
            lr=encoder_lr
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.encoder.train(training)
        self.FFD.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

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

    def update_critic(self, aug_obs, action, reward, discount, aug_next_obs,
                               step, strong_aug_obs, strong_aug_move_obs):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(aug_next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(aug_next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(aug_obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        aug_Q1, aug_Q2 = self.critic(strong_aug_obs, action)
        strong_aug_loss = F.mse_loss(aug_Q1, target_Q) + F.mse_loss(aug_Q2, target_Q)

        #  TODO
        if step > self.aux_latency:
            strong_aug_move_obs_Q1, strong_aug_move_obs_Q2 = self.critic(strong_aug_move_obs, action)
            strong_aug_move_loss = F.mse_loss(strong_aug_move_obs_Q1, target_Q) + F.mse_loss(strong_aug_move_obs_Q2, target_Q)
            critic_loss = 0.5 * critic_loss + 0.25 * (strong_aug_loss + strong_aug_move_loss)
        else:
            critic_loss = 0.5 * (critic_loss + strong_aug_loss)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # Optimize the critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.encoder_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.encoder_optimizer.step()

        if self.use_tb:  # TODO
            grad_conv1 = self.encoder.third_cnn.model.conv1.weight.grad
            metrics['grad_conv1_critic_mean'] = grad_conv1.mean().item() if grad_conv1 is not None else 0
            metrics['grad_conv1_critic_max'] = grad_conv1.max().item() if grad_conv1 is not None else 0
            metrics['grad_conv1_critic_min'] = grad_conv1.min().item() if grad_conv1 is not None else 0
        return metrics

    def update_actor_and_alpha(self, aug_obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(aug_obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(aug_obs, action)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha.detach() * log_prob -Q).mean()

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        grad = self.encoder.third_cnn.model.conv1.weight.grad

        self.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['alpha_loss'] = alpha_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_auxiliary(self, step, fix_obs, move_obs, ego_obs):  #  TODO
        metrics = dict()

        # update auxiliary task
        def calc_aux():
            z_a = self.FFD.encode(fix_obs, ego_obs)
            z_pos = self.FFD.encode(move_obs, ego_obs, ema=True)
            logits = self.FFD.compute_logits(z_a, z_pos)
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            contrastive_loss = self.cross_entropy_loss(logits, labels)
            loss = contrastive_loss

            if self.use_tb:
                metrics['aux_contrastive_loss'] = contrastive_loss.item()

            return loss

        if step > self.aux_latency:
            self.FFD_optimizer.zero_grad(set_to_none=True)
            loss = calc_aux()
            loss.backward()
            self.FFD_optimizer.step()
        else:
            metrics['aux_contrastive_loss'] = 0
            metrics['aux_lr'] = self.FFD_optimizer.param_groups[0]['lr']

        if self.use_tb:  # TODO
            grad = self.encoder.third_cnn.model.conv1.weight.grad
            metrics['grad_aux_mean'] = grad.mean().item() if grad is not None else 0
            metrics['grad_aux_max'] = grad.max().item() if grad is not None else 0
            metrics['grad_aux_min'] = grad.min().item() if grad is not None else 0

        return metrics

    def update(self, replay_iter, step, traj):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # auxiliary
        l = obs.shape[1] // 3
        fix_obs = obs.float()[:, :l]
        move_obs = obs.float()[:, l:2*l]
        ego_obs = obs.float()[:, 2*l:3*l]

        fix_next_obs = next_obs.float()[:, :l]
        ego_next_obs = next_obs.float()[:, 2*l:3*l]

        # augment
        aug_fix_obs = self.aug(fix_obs)
        aug_ego_obs = self.aug(ego_obs)
        aug_fix_next_obs = self.aug(fix_next_obs)
        aug_ego_next_obs = self.aug(ego_next_obs)

        # strong augmentation
        original_fix_obs = aug_fix_obs.clone()
        original_ego_obs = aug_ego_obs.clone()
        original_move_obs = move_obs.clone()
        if l % 3 == 0:
            strong_aug_fix_obs = random_mask_freq_v2(random_overlay(original_fix_obs))
            strong_aug_ego_obs = random_mask_freq_v2(random_overlay(original_ego_obs))
            if step > self.aux_latency:
                strong_aug_move_obs = random_mask_freq_v2(random_overlay(original_move_obs))
                strong_aug_move_obs = self.encoder(strong_aug_move_obs, strong_aug_ego_obs)
            else:
                strong_aug_move_obs = None
        else:
            strong_aug_fix_obs = random_mask_freq_v2(random_overlay(original_fix_obs[:, :l-1]))
            strong_aug_ego_obs = random_mask_freq_v2(random_overlay(original_ego_obs[:, :l-1]))
            strong_aug_fix_obs = torch.cat([strong_aug_fix_obs, original_fix_obs[:, l - 1:l]], dim=1)
            strong_aug_ego_obs = torch.cat([strong_aug_ego_obs, original_ego_obs[:, l - 1:l]], dim=1)
            if step > self.aux_latency:
                strong_aug_move_obs = random_mask_freq_v2(random_overlay(original_move_obs[:, :l-1]))
                strong_aug_move_obs = torch.cat([strong_aug_move_obs, original_move_obs[:, l - 1:l]], dim=1)
                strong_aug_move_obs = self.encoder(strong_aug_move_obs, strong_aug_ego_obs)
            else:
                strong_aug_move_obs = None

        strong_aug_obs = self.encoder(strong_aug_fix_obs, strong_aug_ego_obs)
        aug_obs = self.encoder(aug_fix_obs, aug_ego_obs)
        with torch.no_grad():
            aug_next_obs = self.encoder(aug_fix_next_obs, aug_ego_next_obs)

        # update critic
        metrics.update(
            self.update_critic(aug_obs, action, reward, discount, aug_next_obs,
                               step, strong_aug_obs, strong_aug_move_obs)
        )

        # update actor and alpha
        metrics.update(
            self.update_actor_and_alpha(aug_obs.detach(), step)
        )

        # update auxiliary task
        # TODO
        metrics.update(
            self.update_auxiliary(step, fix_obs, move_obs, ego_obs)
        )

        # update critic target and encoder target
        # TODO
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
