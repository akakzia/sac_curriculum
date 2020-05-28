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
import pickle

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

def compute_dist(o1, o2):
    dists = [np.linalg.norm(o1[i] - o2[i]) for i in range(3)]
    return np.sum(dists)

if __name__ == '__main__':
    num_eval = 50
    path = '/home/flowers/Downloads/test/'
    model_path = path + 'model_600.pt'

    with open(path + 'config.json', 'r') as f:
        params = json.load(f)
    params['symmetry_trick'] = False
    params['small_deepset'] = True
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
    rollout_worker = RolloutWorker(env, policy, goal_sampler, args)

    eval_goals = goal_sampler.valid_goals
    inits = [None] * len(eval_goals)
    all_results = []

    cont_env = gym.make('FetchManipulate3ObjectsContinuous-v0')
    continuous_sample = cont_env.unwrapped.sample_continuous_goal_from_binary_goal
    all_pos = np.zeros([num_eval, len(eval_goals), 3 * 3, 3])
    dists = np.zeros([num_eval, len(eval_goals), 2])
    for i in range(num_eval):
        print(i)
        for i_eg, eg in enumerate(eval_goals):
            success = False
            while not success:
                episodes = rollout_worker.generate_rollout(inits, eg.reshape(1, -1), self_eval=True, true_eval=True, animated=True)
                if str(episodes[0]['g'][0]) == str(episodes[0]['ag'][-1]):
                    success = True
            o_init = episodes[0]['obs'][0]
            obj_init = [o_init[10 + 15 * i: 10 + 3 + 15 * i] for i in range(3)]
            o_final = episodes[0]['obs'][-1]
            obj_final = [o_final[10 + 15 * i: 10 + 3 + 15 * i] for i in range(3)]
            obj_lanier = [continuous_sample(eg)[i * 3: 3 * (i+ 1)] for i in range(3)]
            dist_our = compute_dist(obj_init, obj_final)
            dist_lanier = compute_dist(obj_init, obj_lanier)
            all_pos[i, i_eg, :, 0] = np.array(obj_init).flatten()
            all_pos[i, i_eg, :, 1] = np.array(obj_final).flatten()
            all_pos[i, i_eg, :, 2] = np.array(obj_lanier).flatten()
            dists[i, i_eg, 0] = dist_our
            dists[i, i_eg, 1] = dist_lanier

        #
        # results = np.array([str(e['g'][0]) == str(e['ag'][-1]) for e in episodes]).astype(np.int)
        # all_results.append(results)

    with open('/home/flowers/Desktop/dist_pos.pkl', 'wb') as f:
        pickle.dump(all_pos, f)

    with open('/home/flowers/Desktop/dist.pkl', 'wb') as f:
        pickle.dump(dists, f)
    # results = np.array(all_results)
    # print('Av Success Rate: {}'.format(results.mean()))