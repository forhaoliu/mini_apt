import numpy as np
from utils.atari_env import make_env
from utils.replay_buffer import Buffer


# finetuning last layers of Q function given downstream reward
def evaluation(algo, eval_algo, args):
    eval_env = make_env(args)
    eval_env.seed(args.seed)
    buffer = Buffer(args)
    total_timestep = 0
    for k in range(args.eval_num):
        state, done = eval_env.reset(), False
        episode_timesteps = 0
        while not done and episode_timesteps < args.max_epilen:
            episode_timesteps += 1
            action = algo.select_action(np.array(state), stochastic=True)
            next_state, reward, done, _ = eval_env.step(action)
            not_done = 1.0 - float(done) if episode_timesteps < args.max_epilen else 1.0
            buffer.add(state.cpu(), action, next_state.cpu(), reward, not_done)
            state = next_state
            if total_timestep >= args.batch_size:
                eval_algo.train(buffer, args)
            if done or episode_timesteps >= args.max_epilen:
                state, done = eval_env.reset(), False

    # testing after finetuning
    eval_env = make_env(args, evaluate=True)
    eval_env.seed(args.seed + 100)
    reward_sum = 0
    for k in range(args.eval_num):
        state, done = eval_env.reset(), False
        episode_timesteps = 0
        while not done and episode_timesteps < args.max_epilen:
            episode_timesteps += 1
            action = algo.select_action(np.array(state), stochastic=False)
            next_state, reward, done, _ = eval_env.step(action)
            not_done = 1.0 - float(done) if episode_timesteps < args.max_epilen else 1.0
            state = next_state
            reward_sum += reward
            if done or episode_timesteps >= args.max_epilen:
                state, done = eval_env.reset(), False
    return reward_sum / args.eval_num
