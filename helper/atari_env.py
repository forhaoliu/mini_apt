import collections
import random

import atari_py
import cv2
import torch


class AtariEnv:
    def __init__(self, args):
        self.ale = atari_py.ALEInterface()
        self.seed(args.seed)
        self.ale.setInt("max_num_frames_per_episode", args.max_epilen)
        self.ale.setInt("frame_skip", 0)
        self.ale.setBool("color_averaging", False)
        self.ale.setFloat("repeat_action_probability", 0)
        self.ale.loadROM(atari_py.get_game_path(args.env_name))  # required
        self.action_set = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(self.action_set)), self.action_set))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.loss_of_life = False  # check if reset due to loss of life
        self.frame_stack = args.frame_stack
        self.state_buffer = collections.deque([], maxlen=args.frame_stack)
        self.training = True

    def seed(self, seed):
        self.ale.setInt("random_seed", seed)

    def train(self):
        self.training = True  # loss of life as terminal signal

    def eval(self):
        self.training = False  # standard terminal signal

    def render(self):
        return self.ale.getScreenRGB2()

    def close(self):
        cv2.destroyAllWindows()

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32).div_(255)

    def _reset_buffer(self):
        for _ in range(self.frame_stack):
            self.state_buffer.append(torch.zeros(84, 84))

    def reset(self):
        if self.loss_of_life:
            self.loss_of_life = False
            self.ale.act(0)  # use no-op after loss of life
        else:
            self._reset_buffer()
            self.ale.reset_game()
            for _ in range(random.randrange(30)):
                self.ale.act(0)
                if self.ale.game_over():
                    self.ale.reset_game()
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):  # repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action.item()))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # lives > 0 for Q*bert
                self.loss_of_life = not done  # set loss of life flag when not truly done
                done = True
            self.lives = lives
        return torch.stack(list(self.state_buffer), 0), reward, done, done

    @property
    def action_num(self):
        return len(self.actions)


def make_env(args, evaluate=False):
    env = AtariEnv(args)
    if evaluate:
        env.eval()
    else:
        env.train()
    return env
