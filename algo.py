import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
from entropy import compute_apt_reward
from utils.utils import grad_false, hard_update, soft_update
from model import Critic, RndEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Algo(object):
    def __init__(self, args):
        self.critic = Critic(args).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = Critic(args).to(device).eval()
        hard_update(self.critic_target, self.critic)
        grad_false(self.critic_target)
        self.state_encoder = RndEncoder(args).to(device) if args.enc == "random" else self.critic.encoder
        self.action_range = args.action_range
        self.total_it = 0

    @torch.no_grad()
    def select_action(self, state, stochastic, epsilon=0.001):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        if stochastic and np.random.rand() < epsilon:
            action = np.array([np.random.randint(self.action_range[0], self.action_range[1])])
        else:
            current_Q = self.critic(state)
            action = torch.argmax(current_Q, -1).long()
            action = action.clamp(*self.action_range).cpu().data.numpy().flatten()
        return action

    @torch.no_grad()
    def compute_apt_reward(self, state, action, next_state, reward, not_done, next_state_2, args):
        source = self.state_encoder(next_state)
        target = self.state_encoder(next_state_2)
        reward = compute_apt_reward(source, target, args)
        return reward

    def train(self, buffer, args):
        self.total_it += 1

        all_critic_loss = []
        state, action, next_state, reward, not_done, next_state_2 = buffer.sample(args.batch_size)
        reward = self.compute_apt_reward(state, action, next_state, reward, not_done, next_state_2, args)

        with torch.no_grad():
            next_action = self.critic(next_state).argmax(-1).view(-1, 1).long()  # (b, a) -> (b, 1)
            target_Q = self.critic_target(next_state).gather(-1, next_action)  # (b, 1)
            target_Q = reward + (not_done * args.gamma * target_Q)  # (b, 1) + (b, 1)
            target_Q_1, target_Q_2 = torch.split(target_Q, args.batch_size, dim=0)
            target_Q = (target_Q_1 + target_Q_2) / 2.0

        current_Q = self.critic(state).gather(-1, action)  # (b, a) -> (b, 1)
        current_Q_1, current_Q_2 = torch.split(current_Q, args.batch_size, dim=0)
        critic_loss = F.mse_loss(current_Q_1, target_Q) + F.mse_loss(current_Q_2, target_Q)

        all_critic_loss.append(critic_loss.data.item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % args.target_update_freq == 0:
            soft_update(self.critic_target, self.critic, args.tau)

    def train_repr(self, state):
        # todo: port tf representation learning code here
        pass

    def save(self, model_dir, t, args):
        torch.save(self.critic.state_dict(), os.path.join(model_dir, f"critic_{t}.pth"))

    def load(self, model_dir, t, args):
        self.critic.load_state_dict(torch.load(os.path.join(model_dir, f"critic_{t}.pth")))
        self.critic_target = copy.deepcopy(self.critic)


class EvalAlgo(Algo):
    def __init__(self, algo, args):
        super().__init__(args)
        self.critic = Critic(args).to(device)

        # load parameters from algo.critic
        self.critic.load_state_dict(algo.critic.state_dict())

        # only finetune the last few layers
        self.critic_optimizer = torch.optim.Adam(self.critic.fc.parameters(), lr=3e-4)

        self.critic_target = Critic(args).to(device).eval()
        hard_update(self.critic_target, self.critic)
        grad_false(self.critic_target)
        self.state_encoder = RndEncoder(args) if args.enc == "rnd" else self.critic.encoder
        self.action_range = args.action_range
        self.total_it = 0
