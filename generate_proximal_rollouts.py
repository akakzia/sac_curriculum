import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from rollout import RolloutWorker
from goal_sampler import GoalSampler
import random
from mpi4py import MPI
from arguments import get_args
import proximal_generator.gvae as gvae

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params


def do_rollout(policy, env, pgg, animated=False):
    observation = env.unwrapped.reset()
    obs = observation['observation']
    ag = observation['achieved_goal']
    counter = 0
    while counter < 50:
        ag_transformed = np.expand_dims(np.clip(ag + 1, 0, 1), 0)
        reached = False
        trial_counter = 0
        while trial_counter < 10:
            index = np.random.randint(5)
            g_transformed = (pgg.inference(None, torch.Tensor(ag_transformed), n=1, index=[index]).detach().numpy() > 0.5).astype(np.float32)
            g = g_transformed.copy().squeeze(0)
            g[np.where(g == 0)] = -1
            env.unwrapped.target_goal = np.array(g)
            env.unwrapped.binary_goal = np.array(g)

            for t in range(40):
                action = policy.act(obs.copy(), ag.copy(), g.copy(), None, True)
                if animated:
                    env.render()
                observation_new, r, _, _ = env.step(action)
                obs = observation_new['observation']
                ag = observation_new['achieved_goal']
            if r == 1.:
                reached = True
                counter += 1
                break
            else:
                trial_counter += 1
        if not reached:
            break
    print('Counter: {}'.format(counter))
    return counter

if __name__ == '__main__':
    fix_permutation = True
    num_eval = 10
    path = '/home/ahmed/'
    model_path = path + 'model_160.pt'
    save_path = 'trajectories_5blocks_reduced.pkl'

    args = get_args()

    args.env_name = 'FetchManipulate{}Objects-v0'.format(args.n_blocks)

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

    # def goal generator
    with open('proximal_generator/data/vae_model2.pkl', 'rb') as f:
        pgg = torch.load(f)
    all_results = []
    for i in range(num_eval):
        maximum_len = do_rollout(policy, env, pgg, animated=False)
        all_results.append(maximum_len)

    print('Average sequence of goals before failure: {}'.format(np.mean(all_results)))
    print('Standard deviation: {}'.format(np.std(all_results)))