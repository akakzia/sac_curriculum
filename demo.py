import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
import  random
from mpi4py import MPI
from language.build_dataset import sentence_from_configuration
from utils import get_instruction, get_eval_goals
from arguments import get_args
import pickle as pkl
from itertools import combinations, permutations

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'num_objects': env.num_blocks, 'max_timesteps': env._max_episode_steps}
    return params


# def get_eval_goals(close_pairs, stack_size=1, nb_goals=1):
#     """ Return the eval goals that respect the close_pairs and above size"""
#     res = []
#     if stack_size == 1:
#         ids = []
#         for _ in range(nb_goals):
#             id = np.random.choice(np.arange(10), size=close_pairs, replace=False)
#             ids.append(id)
#         for id in ids:
#             g = np.zeros(30)
#             g[id] = 1.
#             res.append(g)
#         return np.array(res)
#     else:
#         ids = []
#         for _ in range(nb_goals):
#             id = []
#             objects = np.random.choice(np.arange(5), size=stack_size, replace=False)
#             for j in range(stack_size-1):
#                 obj_ids = (objects[j], objects[j+1])
#                 for k, c in enumerate(combinations([0, 1, 2, 3, 4], 2)):
#                     if set(obj_ids) == set(c):
#                         id.append(k)
#                 for k, c in enumerate(permutations([0, 1, 2, 3, 4], 2)):
#                     if obj_ids == c:
#                         id.append(k+10)
#             ids.append(np.array(id))
#         for id in ids:
#             g = np.zeros(30)
#             g[id] = 1.
#             res.append(g)
#         return np.array(res)



if __name__ == '__main__':
    num_eval = 1
    path = '/home/akakzia/'
    model_path = path + 'model_570.pt'

    # with open(path + 'config.json', 'r') as f:
    #     params = json.load(f)
    # args = SimpleNamespace(**params)
    args = get_args()

    if args.algo == 'continuous':
        args.env_name = 'FetchManipulate3ObjectsContinuous-v0'
        args.multi_criteria_her = True
    else:
        args.env_name = 'FetchManipulate5Objects-v0'

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
        policy = RLAgent(args, env.compute_reward, goal_sampler)
        policy.load(model_path, args)
    else:
        raise NotImplementedError

    # def rollout worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    # eval_goals = goal_sampler.valid_goals
    eval_goals = get_eval_goals('mixed_3_2', nb_goals=10)
    if args.algo == 'language':
        language_goal = get_instruction()
        eval_goals = np.array([goal_sampler.valid_goals[0] for _ in range(len(language_goal))])
    else:
        language_goal = None
    inits = [None] * len(eval_goals)
    all_results = []
    for i in range(num_eval):
        episodes = rollout_worker.generate_rollout(eval_goals, self_eval=True, true_eval=True, animated=True, language_goal=language_goal)
        if args.algo == 'language':
            results = np.array([e['language_goal'] in sentence_from_configuration(e['ag'][-1], all=True) for e in episodes]).astype(np.int)
        elif args.algo == 'continuous':
            results = np.array([e['rewards'][-1] == 3. for e in episodes])
        else:
            results = np.array([str(e['g'][0]) == str(e['ag'][-1]) for e in episodes]).astype(np.int)
        all_results.append(results)

    results = np.array(all_results)
    print('Av Success Rate: {}'.format(results.mean()))