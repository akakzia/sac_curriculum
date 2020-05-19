import torch
from rl_modules.sac_agent2 import SACAgent
from arguments import get_args
import env
import gym
import numpy as np
from utils import generate_goals
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
import  random
from mpi4py import MPI
import os
import pickle as pkl


# process the inputs
def normalize_goal(g, g_mean, g_std, args):
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    return g_norm


def normalize(o, o_mean, o_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    return o_norm


def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    return inputs


def get_env_params(env):
    obs = env.reset()

    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params


def get_state(obs):
    coordinates_list = [obs[10 + 15*i:13 + 15*i] for i in range(3)]
    return np.concatenate(coordinates_list)


if __name__ == '__main__':
    num_eval = 2
    path = '/home/flowers/Downloads/test/'
    model_path = path + 'model_600.pt'

    with open(path + 'config.json', 'r') as f:
        params = json.load(f)
    args = SimpleNamespace(**params)

    # Make the environment
    env = gym.make(args.env_name)

    # set random seeds for reproduce
    args.seed = np.random.randint(1e6)
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    args.env_params = get_env_params(env)

    goal_sampler = GoalSampler(args)

    # create the sac agent to interact with the environment
    if args.agent == "SAC":
        policy = SACAgent(args, env.compute_reward, goal_sampler)
        policy.load(model_path, args)
    else:
        raise NotImplementedError

    # def rollout worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    # eval_goals = goal_sampler.valid_goals
    eval_idxs = np.random.choice(range(len(goal_sampler.valid_goals)), size=num_eval)
    eval_goals = goal_sampler.valid_goals[eval_idxs]
    inits = [None] * len(eval_goals)
    all_results = []
    # for i in range(num_eval):
    episodes = rollout_worker.generate_rollout(inits, eval_goals, self_eval=True, true_eval=True, animated=False)
    results = np.array([str(e['g'][0]) == str(e['ag'][-1]) for e in episodes]).astype(np.int)
    all_results.append(results)
    data = [[e['ag'][0], e['ag'][-1], get_state(e['obs'][0]), get_state(e['obs'][-1])] for e in episodes]

    results = np.array(all_results)
    print('{} - Av Success Rate: {}'.format(MPI.COMM_WORLD.Get_rank(), results.mean()))
    data = sum(MPI.COMM_WORLD.allgather(data), [])
    if MPI.COMM_WORLD.Get_rank() == 0:
        print('Shape of data is {}'.format(np.array(data).shape))
        with open(os.path.join('data_samples.pkl'), 'wb') as f:
            pkl.dump(np.array(data), f)