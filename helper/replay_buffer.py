import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import collections
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimAtariAug:
    def __init__(self, args):
        self.aug = nn.Sequential(nn.ReplicationPad2d(args.image_pad), transforms.RandomCrop((args.state_dim[-1], args.state_dim[-1])))

    def __call__(self, image):
        return self.aug(image)


class Buffer(object):
    def __init__(self, args):
        self.max_size = args.max_buffer_size
        self.n_step = args.n_step
        self.gamma = args.gamma
        self.memory = collections.deque(maxlen=self.max_size)
        self.n_step_buffer = collections.deque(maxlen=self.n_step)
        self.sim_aug = SimAtariAug(args) if args.aug else nn.Identity()

    def add(self, state, action, next_state, reward, not_done):
        self.n_step_buffer.append((state, action, next_state, reward, not_done))
        if len(self.n_step_buffer) < self.n_step:
            return
        next_state, reward, not_done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            n_s, r, n_d = transition[-3:]
            reward = r + self.gamma * reward * n_d
            next_state, not_done = (next_state, not_done) if n_d else (n_s, n_d)
        state, action = self.n_step_buffer[0][:2]
        self.memory.append([state, action, next_state, reward, not_done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, next_state, reward, not_done = zip(*batch)

        state = torch.FloatTensor(np.tile(np.stack(state, 0), (2, 1, 1, 1))).to(device)
        action = torch.LongTensor(np.tile(np.stack(action, 0), (2, 1))).to(device)
        next_state = torch.FloatTensor(np.tile(np.stack(next_state, 0), (2, 1, 1, 1))).to(device)
        reward = torch.FloatTensor(np.tile(np.array(reward), (2,))).unsqueeze(1).to(device)
        not_done = torch.FloatTensor(np.tile(np.array(not_done), (2,))).unsqueeze(1).to(device)
        return self.sim_aug(state), action, self.sim_aug(next_state), reward, not_done

    def __len__(self):
        return len(self.memory)
