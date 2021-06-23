import argparse
import collections
import json
import os
from datetime import datetime
from types import SimpleNamespace

import atari_py
import numpy as np
import torch
from tqdm import trange
from utils.atari_env import make_env
from algo import Algo, EvalAlgo
from utils.utils import set_seed
from utils.replay_buffer import Buffer
from utils.evaluation import evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_size", type=int, default=int(1e5))
    parser.add_argument("--enable_cudnn", action="store_true")
    parser.add_argument("--env_name", type=str, default="space_invaders", choices=atari_py.list_games())
    parser.add_argument("--id", type=str, default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--start_timesteps", type=int, default=int(20e3))
    parser.add_argument("--critic_hs", type=int, default=512)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--n_step", default=1, type=int)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--target_update_freq", type=int, default=int(8e3))
    parser.add_argument("--model_dir", default="", type=str)
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--max_epilen", type=int, default=int(108e3))
    # visual part
    parser.add_argument("--action_repeat", default=4, type=int)
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--image_pad", type=int, default=4)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--aug", action="store_true")
    # entropy
    parser.add_argument("--knn_avg", action="store_true")
    parser.add_argument("--rms", action="store_true")
    parser.add_argument("--knn_k", type=int, default=16)
    parser.add_argument("--knn_bs", default=2048, type=int)
    parser.add_argument("--knn_clip", default=0.0005, type=float)
    parser.add_argument("--tgt", default="same", type=str, choices=["same", "neg"])
    parser.add_argument("--enc", default="random", type=str, choices=["random", "contrastive"])
    parser.add_argument("--embed_os", type=int, default=256)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = args.enable_cudnn

    defaults = {
        "max_timesteps": int(50e6) + 1,
        "gamma": 0.99,
        "tau": 0.01,
        "grad_clip": None,
        "log_freq": int(5e3),
        "eval_num": 10,
    }
    defaults.update(vars(args))
    args = SimpleNamespace(**defaults)

    set_seed(args)
    env = make_env(args)
    env.seed(args.seed)
    args.state_dim = (args.frame_stack, args.image_size, args.image_size)
    args.action_num = env.action_num
    args.action_range = [0, args.action_num - 1]

    args.log_dir = os.path.join("logs", args.env_name, f'{args.id}-seed={args.seed}-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(args.log_dir, exist_ok=True)
    with open(f"{args.log_dir}/config.json", "w") as fp:
        json.dump(collections.OrderedDict(vars(args)), fp)

    algo = Algo(args)

    if args.model_dir != "":
        algo.load(args.model_dir, args.model_id, args)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    eval_nums = 0

    model_dir = os.path.join(args.log_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    buffer = Buffer(args)

    for t in trange(args.max_timesteps):
        episode_timesteps += 1

        if t < args.start_timesteps:
            action = np.array([np.random.randint(args.action_range[0], args.action_range[1])])
        else:
            action = algo.select_action(np.array(state), stochastic=True)

        next_state, reward, done, _ = env.step(action)
        not_done = 1.0 - float(done) if episode_timesteps < args.max_epilen else 1.0

        buffer.add(state.cpu(), action, next_state.cpu(), reward, not_done)

        state = next_state
        episode_reward += reward

        if t % args.train_freq == 0 and t >= args.start_timesteps:
            algo.train(buffer, args)

        if done or episode_timesteps >= args.max_epilen:
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if t % args.log_freq == 0 and t >= args.start_timesteps:
            eval_nums += 1
            eval_algo = EvalAlgo(algo, args)
            eval_return = evaluation(algo, eval_algo, args)
            print(f"pretraining steps are {t}, evaluated return is {eval_return}")
            algo.save(model_dir, t + 1, args)
