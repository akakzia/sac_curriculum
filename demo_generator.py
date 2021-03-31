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

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': 20}
    return params

if __name__ == '__main__':
    num_eval = 20
    path = '/home/akakzia/'
    model_path = path + 'gangstr_sp.pt'
    model_path2 = path + 'gangstr_no_sp.pt'

    # with open(path + 'config.json', 'r') as f:
    #     params = json.load(f)
    # args = SimpleNamespace(**params)
    args = get_args()

    if args.algo == 'continuous':
        args.env_name = 'FetchManipulate{}ObjectsContinuous-v0'.format(args.n_blocks)
        args.multi_criteria_her = True
    else:
        args.env_name = 'FetchManipulate{}Objects-v0'.format(args.n_blocks)

    # Make the environment
    env = gym.make(args.env_name)

    # set random seeds for reproduce
    args.seed = 10041
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
        policy2 = RLAgent(args, env.compute_reward, goal_sampler)
        policy2.load(model_path2, args)
    else:
        raise NotImplementedError

    # def rollout worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)
    rollout_worker2 = RolloutWorker(env, policy2, goal_sampler, args)

    # eval_goals = goal_sampler.valid_goals
    # eval_goals, eval_masks = goal_sampler.generate_eval_goals()
    eval_goals = []
    for i in range(num_eval):
        instruction = 'stack_4'
        eval_goal = get_eval_goals(instruction, n=args.n_blocks)
        eval_goals.append(eval_goal.squeeze(0))
    eval_masks = np.zeros((num_eval, 18))
    eval_goals = np.array(eval_goals)
    if args.algo == 'language':
        language_goal = get_instruction()
        eval_goals = np.array([goal_sampler.valid_goals[0] for _ in range(len(language_goal))])
    else:
        language_goal = None
    inits = [None] * len(eval_goals)
    all_results = []
    all_results2 = []

    episodes = rollout_worker.generate_rollout(eval_goals, eval_masks, self_eval=True, true_eval=True, animated=False, language_goal=language_goal)
    episodes2 = rollout_worker2.generate_rollout(eval_goals, eval_masks, self_eval=True, true_eval=True, animated=False, language_goal=language_goal)
    results = np.array([e['success'][-1].astype(np.float32) for e in episodes])
    results2 = np.array([e['success'][-1].astype(np.float32) for e in episodes2])
    all_results.append(results)
    all_results2.append(results2)

    results = np.array(all_results)
    results2 = np.array(all_results2)
    print('Av Success Rate: {}'.format(results.mean()))
    print('Av Success Rate: {}'.format(results2.mean()))

    # with open('../BtoL.pkl', 'wb') as f:
    #     pkl.dump(episodes, f)
