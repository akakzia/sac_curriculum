import torch
from rl_modules.sac_agent import SACAgent
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

if __name__ == '__main__':
    num_eval = 100
    path = '/home/ahakakzia/DECSTR/models/'
    model_path = path + 'model_320.pt'

    # with open(path + 'config.json', 'r') as f:
    #     params = json.load(f)
    # args = SimpleNamespace(**params)
    args = get_args()

    # Make the environment
    env = gym.make(args.env_name)

    # set random seeds for reproduce
    # args.seed = np.random.randint(1e6)
    # rs = np.random.RandomState(seed=42)
    with open('../random_state.pkl', 'rb') as f:
        rs = pkl.load(f)
    seed = rs.randint(1e6)
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

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
    # buckets = np.random.choice(np.arange(5), size=2)
    # eval_goals = np.array([goal_sampler.oracle_id_to_g[np.random.choice(goal_sampler.buckets[b])] for b in buckets])
    eval_goals = goal_sampler.valid_goals
    inits = [None] * len(eval_goals)
    all_results = []
    all_episodes = []
    for i in range(num_eval):
        episodes = rollout_worker.generate_rollout(eval_goals, self_eval=True, true_eval=True, animated=False)
        if MPI.rank == 0:
            print(i)
        if args.algo == 'continuous':
            results = np.array([e['rewards'][-1] == 3. for e in episodes]).astype(np.int)
        else:
            results = np.array([str(e['g_binary'][0]) == str(e['ag_binary'][-1]) for e in episodes]).astype(np.int)
        all_results.append(results)
        all_episodes.append(episodes)

    results = np.array(all_results)
    all_episodes = np.array(all_episodes)
    all_episodes = MPI.COMM_WORLD.gather(all_episodes, root=0)
    # for e, b in zip(episodes, buckets):
    #     e['bucket'] = b
    if MPI.rank == 0:
        print('Av Success Rate: {}'.format(results.mean()))
        with open('../rebuttal_continuous_final.pkl', 'wb') as f:
            pkl.dump(all_episodes, f)