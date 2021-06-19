import numpy as np
import collections
import torch.nn as nn
import kornia
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Buffer(object):
    def __init__(self, args):
        self.ptr = 0
        self.size = 0
        self.max_size = args.max_size
        self.knn_bs = args.knn_bs
        self.tgt = args.tgt

        self.aug = (
            nn.Sequential(nn.ReplicationPad2d(args.image_pad), kornia.augmentation.RandomCrop((args.state_dim[-1], args.state_dim[-1])))
            if args.aug
            else nn.Identity()
        )

        self.state = np.empty((args.max_size, *args.state_dim), dtype=np.uint8)
        self.action = np.empty((args.max_size, 1), dtype=np.uint8)
        self.next_state = np.empty((args.max_size, *args.state_dim), dtype=np.uint8)
        self.reward = np.empty((args.max_size, 1), dtype=np.float32)
        self.not_done = np.empty((args.max_size, 1), dtype=np.float32)

        # N-step return (reward)
        self.n_step_buffer = collections.deque(maxlen=args.n_step)
        self.n_step = args.n_step
        self.gamma = args.gamma

    def add(self, state, action, next_state, reward, not_done):
        self.n_step_buffer.append((state, action, next_state, reward, not_done))
        # if n steps are not ready
        if len(self.n_step_buffer) < self.n_step:
            return
        # make a n-step transition
        next_state, reward, not_done = self.n_step_buffer[-1][-3:]  # last transition
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            n_s, r, n_d = transition[-3:]
            reward = r + self.gamma * reward * n_d
            next_state, not_done = (next_state, not_done) if n_d else (n_s, n_d)
        state, action = self.n_step_buffer[0][:2]

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = not_done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if self.tgt == "same":
            ind = np.random.choice(self.size, size=batch_size) if batch_size <= self.size else np.arange(self.size)
            tgt_ind = ind
        elif self.tgt == "neg":
            ind = np.random.choice(np.arange(self.size), size=batch_size) if batch_size <= self.size else np.arange(self.size)
            tgt_ind = np.arange(self.size)
            mask = np.ones_like(tgt_ind)
            mask[ind] = False
            tgt_ind = tgt_ind[mask]
            knn_bs = self.knn_bs if self.knn_bs > 0 else tgt_ind.shape[0]  #  use all the rest of replay buffer
            if knn_bs < tgt_ind.shape[0]:
                tgt_ind = tgt_ind[np.random.choice(tgt_ind.shape[0], knn_bs)]
        return (
            self.aug(torch.FloatTensor(np.tile(self.state[ind], (2, 1, 1, 1))).to(device)),
            torch.LongTensor(np.tile(self.action[ind], (2, 1))).to(device),
            self.aug(torch.FloatTensor(np.tile(self.next_state[ind], (2, 1, 1, 1))).to(device)),
            torch.FloatTensor(np.tile(self.reward[ind], (2, 1))).to(device),
            torch.FloatTensor(np.tile(self.not_done[ind], (2, 1))).to(device),
            self.aug(torch.FloatTensor(np.tile(self.next_state[tgt_ind], (2, 1, 1, 1))).to(device)),
        )
