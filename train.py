import argparse
import collections
import json
import os
from datetime import datetime

import atari_py
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import helper.utils as utils
from algo import Algo, EvalAlgo
from helper.atari_env import make_env
from helper.evaluation import evaluation
from helper.replay_buffer import Buffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_buffer_size", type=int, default=int(1e6))
    parser.add_argument("--max_timestep", type=int, default=int(1e8) + 1)
    parser.add_argument("--log_freq", type=int, default=int(100e5))
    parser.add_argument("--eval_num", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--enable_cudnn", action="store_true")
    parser.add_argument("--env_name", type=str, default="space_invaders", choices=atari_py.list_games())
    parser.add_argument("--id", type=str, default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--start_timestep", type=int, default=int(20e3))
    parser.add_argument("--critic_hs", type=int, default=512)
    parser.add_argument("--norm_clip", type=float, default=10)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--target_update_freq", type=int, default=int(8e3))
    parser.add_argument("--model_dir", default="", type=str)
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--max_epilen", type=int, default=int(108e3))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n_step", default=3, type=int)
    # visual part
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--image_pad", type=int, default=4)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--rainbow_conv", action="store_true")
    parser.add_argument("--proj_dim", type=int, default=256)
    # entropy
    parser.add_argument("--knn_avg", action="store_true")
    parser.add_argument("--knn_rms", action="store_true")
    parser.add_argument("--knn_k", type=int, default=16)
    parser.add_argument("--knn_clip", type=float, default=0.0005, help="-1 to disable")
    # finetune
    # hps should follow https://github.com/Kaixhin/Rainbow/blob/master/main.py
    parser.add_argument("--reward_clip", type=int, default=1)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = args.enable_cudnn

    utils.set_seed(args)
    env = make_env(args)
    env.seed(args.seed)
    args.state_dim = (args.frame_stack, args.image_size, args.image_size)
    args.action_num = env.action_num
    args.action_range = [0, args.action_num - 1]

    args.log_dir = os.path.join(
        "logs" if not args.debug else "/tmp/", args.env_name, f'{args.id}-seed={args.seed}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )
    os.makedirs(args.log_dir, exist_ok=True)
    with open(f"{args.log_dir}/config.json", "w") as fp:
        json.dump(collections.OrderedDict(vars(args)), fp)

    writer = args.writer = SummaryWriter(args.log_dir)

    algo = Algo(args)

    if args.model_dir != "":
        algo.load(args.model_dir, args.model_id, args)

    state, done = env.reset(), False
    episode_timestep = 0
    episode_num = 0
    eval_nums = 0

    model_dir = os.path.join(args.log_dir, "model")
    vis_dir = os.path.join(args.log_dir, "vis")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    buffer = Buffer(args)

    for t in trange(args.max_timestep):
        episode_timestep += 1

        if t < args.start_timestep:
            action = np.array([np.random.randint(args.action_range[0], args.action_range[1])])
        else:
            action = algo.select_action(np.array(state), stochastic=True)

        next_state, reward, done, _ = env.step(action)
        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip)
        not_done = 1.0 - float(done) if episode_timestep < args.max_epilen else 1.0

        buffer.add(state, action, next_state, reward, not_done)

        state = next_state

        if t % args.train_freq == 0 and t >= args.start_timestep:
            algo.train(buffer, args)
            writer.add_scalar("Train/total_timesteps", t, algo.total_it)
            writer.flush()

        if done or episode_timestep >= args.max_epilen:
            state, done = env.reset(), False
            episode_timestep = 0
            episode_num += 1

        if t % args.log_freq == 0 and t >= args.start_timestep:
            eval_nums += 1
            eval_algo = EvalAlgo(algo, args)
            eval_return = evaluation(eval_algo, args)
            writer.add_scalar("Evaluation/episode_reward", eval_return, algo.total_it)
            writer.flush()
            algo.save(model_dir, t + 1, args)
