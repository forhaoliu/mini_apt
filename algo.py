import copy
import os

import numpy as np
import torch
import torch.nn.functional as F

import helper.utils as utils
from entropy import compute_apt_reward
from model import Critic, Encoder
from torch.nn.utils import clip_grad_norm_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Algo(object):
    def __init__(self, args):
        self.state_encoder = Encoder(args).to(device)
        self.repr_optimizer = torch.optim.Adam(self.state_encoder.parameters(), lr=args.lr)
        self.state_encoder_target = Encoder(args).to(device).eval()
        utils.hard_update(self.state_encoder_target, self.state_encoder)
        utils.grad_false(self.state_encoder_target)

        self.critic = Critic(args).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = Critic(args).to(device).eval()
        utils.hard_update(self.critic_target, self.critic)
        utils.grad_false(self.critic_target)

        self.action_range = args.action_range
        self.total_it = 0
        self.writer = args.writer

    @torch.no_grad()
    def select_action(self, state, stochastic, epsilon=0.001):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        if stochastic and np.random.rand() < epsilon:
            action = np.array([np.random.randint(self.action_range[0], self.action_range[1])])
        else:
            current_Q = self.critic(self.state_encoder(state))
            action = torch.argmax(current_Q, -1).long()
            action = action.clamp(*self.action_range).cpu().data.numpy().flatten()
        return action

    @torch.no_grad()
    def compute_apt_reward(self, state, args):
        source = self.state_encoder(state)
        target = self.state_encoder(state)
        reward = compute_apt_reward(source, target, args)
        self.writer.add_scalar("Loss/ent_reward_mean", reward.mean(), self.total_it)
        self.writer.add_histogram("Loss/ent_reward", reward.squeeze(1), self.total_it)
        return reward

    def train(self, buffer, args):
        self.total_it += 1

        all_critic_loss = []
        state, action, next_state, reward, not_done = buffer.sample(args.batch_size)

        # update representation learning
        self.train_repr(state, args)
        utils.soft_update(self.state_encoder_target, self.state_encoder, args.tau)

        # compute entropy max reward
        reward = self.compute_apt_reward(next_state, args)

        with torch.no_grad():
            next_action = self.critic(self.state_encoder(next_state)).argmax(-1).view(-1, 1).long()  # (b, a) -> (b, 1)
            target_Q = self.critic_target(self.state_encoder_target(next_state)).gather(-1, next_action)  # (b, 1)
            target_Q = reward + (not_done * args.gamma * target_Q)  # (b, 1) + (b, 1)
            target_Q_1, target_Q_2 = torch.split(target_Q, args.batch_size, dim=0)
            target_Q = (target_Q_1 + target_Q_2) / 2.0

        current_Q = self.critic(self.state_encoder(state)).gather(-1, action)  # (b, a) -> (b, 1)
        current_Q_1, current_Q_2 = torch.split(current_Q, args.batch_size, dim=0)
        critic_loss = F.mse_loss(current_Q_1, target_Q) + F.mse_loss(current_Q_2, target_Q)

        all_critic_loss.append(critic_loss.data.item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), args.norm_clip)
        self.critic_optimizer.step()

        if self.total_it % args.target_update_freq == 0:
            utils.soft_update(self.critic_target, self.critic, args.tau)

        with torch.no_grad():
            self.writer.add_scalar("Loss/critic_current_Q", current_Q.mean(), self.total_it)
            self.writer.add_scalar("Loss/critic_target_Q", target_Q.mean(), self.total_it)
            self.writer.add_scalar("Loss/critic_loss", critic_loss.detach().item(), self.total_it)

    def train_repr(self, state, args):
        temperature = 0.07
        label = torch.cat([torch.arange(args.batch_size) for i in range(2)], dim=0)
        label = (label.unsqueeze(0) == label.unsqueeze(1)).float().to(device)

        feature = self.state_encoder.project(state)
        feature = F.normalize(feature, dim=1)

        similarity_matrix = torch.matmul(feature, feature.T)

        mask = torch.eye(label.shape[0], dtype=torch.bool).to(device)
        label = label[~mask].view(label.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positive = similarity_matrix[label.bool()].view(label.shape[0], -1)
        negative = similarity_matrix[~label.bool()].view(similarity_matrix.shape[0], -1)

        logit = torch.cat([positive, negative], dim=1) / temperature
        label = torch.zeros(logit.shape[0], dtype=torch.long).to(device)

        ce = torch.nn.CrossEntropyLoss().to(device)
        repr_loss = ce(logit, label)

        self.repr_optimizer.zero_grad()
        repr_loss.backward()
        clip_grad_norm_(self.state_encoder.parameters(), args.norm_clip)
        self.repr_optimizer.step()

        with torch.no_grad():
            self.writer.add_scalar("Loss/repr_loss", repr_loss.detach().item(), self.total_it)

    def save(self, model_dir, t, args):
        torch.save(self.critic.state_dict(), os.path.join(model_dir, f"critic_{t}.pth"))

    def load(self, model_dir, t, args):
        self.critic.load_state_dict(torch.load(os.path.join(model_dir, f"critic_{t}.pth")))
        self.critic_target = copy.deepcopy(self.critic)


class EvalAlgo(Algo):
    def __init__(self, algo, args):
        super().__init__(args)
        self.critic = Critic(args).to(device)
        self.critic.load_state_dict(algo.critic.state_dict())

        self.state_encoder = Encoder(args).to(device)
        self.state_encoder.load_state_dict(algo.state_encoder.state_dict())

        self.critic_target = Critic(args).to(device).eval()
        utils.hard_update(self.critic_target, self.critic)
        utils.grad_false(self.critic_target)

        # finetune the critic but not the encoder
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.action_range = args.action_range
        self.total_it = 0

    def finetune(self, buffer, args):
        self.total_it += 1
        all_critic_loss = []
        state, action, next_state, reward, not_done = buffer.sample(args.batch_size)

        with torch.no_grad():
            next_action = self.critic(self.state_encoder(next_state)).argmax(-1).view(-1, 1).long()  # (b, a) -> (b, 1)
            target_Q = self.critic_target(self.state_encoder_target(next_state)).gather(-1, next_action)  # (b, 1)
            target_Q = reward + (not_done * args.gamma * target_Q)  # (b, 1) + (b, 1)
            target_Q_1, target_Q_2 = torch.split(target_Q, args.batch_size, dim=0)
            target_Q = (target_Q_1 + target_Q_2) / 2.0

        current_Q = self.critic(self.state_encoder(state)).gather(-1, action)  # (b, a) -> (b, 1)
        current_Q_1, current_Q_2 = torch.split(current_Q, args.batch_size, dim=0)
        critic_loss = F.mse_loss(current_Q_1, target_Q) + F.mse_loss(current_Q_2, target_Q)

        all_critic_loss.append(critic_loss.data.item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), args.norm_clip)
        self.critic_optimizer.step()

        if self.total_it % args.target_update_freq == 0:
            utils.soft_update(self.critic_target, self.critic, args.tau)
