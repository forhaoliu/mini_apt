import numpy as np

from helper.atari_env import make_env
from helper.replay_buffer import Buffer


# finetuning last layers of Q function given downstream reward
def evaluation(eval_algo, args):
    eval_env = make_env(args)
    eval_env.seed(args.seed)
    buffer = Buffer(args)
    total_timestep = 0

    max_timestep = int(100e3)
    start_timestep = int(5e3)
    train_freq = 1

    while total_timestep < max_timestep:  # 100K benchmark
        state, done = eval_env.reset(), False
        episode_timesteps = 0
        while not done and episode_timesteps < args.max_epilen:
            episode_timesteps += 1
            total_timestep += 1
            action = eval_algo.select_action(np.array(state), stochastic=True)
            next_state, reward, done, _ = eval_env.step(action)
            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)
            not_done = 1.0 - float(done) if episode_timesteps < args.max_epilen else 1.0
            buffer.add(state, action, next_state, reward, not_done)
            state = next_state

            if total_timestep % train_freq == 0 and total_timestep >= start_timestep:
                eval_algo.finetune(buffer, args)

            if done or episode_timesteps >= args.max_epilen:
                state, done = eval_env.reset(), False

    # testing after finetuning
    eval_env = make_env(args, evaluate=True)
    eval_env.seed(args.seed + 100)
    reward_sum = 0
    max_timestep = int(125e3)
    for k in range(args.eval_num):
        total_timestep = 0
        state, done = eval_env.reset(), False
        episode_timesteps = 0
        while not done and episode_timesteps < args.max_epilen and total_timestep < max_timestep:
            total_timestep += 1
            episode_timesteps += 1
            action = eval_algo.select_action(np.array(state), stochastic=False)
            next_state, reward, done, _ = eval_env.step(action)
            state = next_state
            reward_sum += reward
            if done or episode_timesteps >= args.max_epilen:
                state, done = eval_env.reset(), False
    return reward_sum / args.eval_num
