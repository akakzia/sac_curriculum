import numpy as np
from mpi4py import MPI
import env
import gym
import os, sys
from arguments import get_args
from rl_modules.sac_agent import SACAgent
from rl_modules.td3_agent import TD3Agent
import random
import torch
from rollout import RolloutWorker
from goal_sampler import GoalSampler
from utils import init_storage
import time
from mpi_utils import logger

def get_env_params(env):
    obs = env.reset()

    # close the environment
    # params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
    #           'g_description': obs['goal_description'].shape, 'action': env.action_space.shape[0],
    #           'action_max': env.action_space.high[0], 'max_timesteps': env._max_episode_steps,
    #           'num_blocks': env.num_blocks,
    #           }

    params = {'obs': obs['observation'].shape[0], 'goal_size': env.atomic_goal_size, 'state_description': env.goal_size,
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0], 'max_timesteps': env._max_episode_steps,
              'num_blocks': env.num_blocks, 'num_predicates': env.num_predicates,
              }

    return params

def launch(args):

    rank = MPI.COMM_WORLD.Get_rank()

    t_total_init = time.time()
    # Make the environment
    env = gym.make(args.env_name)

    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # get saving paths
    if rank == 0:
        logdir, model_path, bucket_path = init_storage(args)
        logger.configure(dir=logdir)

    args.env_params = get_env_params(env)

    # def goal sampler:
    goal_sampler = GoalSampler(args)

    # create the sac agent to interact with the environment
    if args.agent == "SAC":
        policy = SACAgent(args, env.compute_reward, goal_sampler)
    elif args.agent == "TD3":
        policy = TD3Agent(args, env.compute_reward, goal_sampler)
    else:
        raise NotImplementedError

    # def rollout worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    # start to collect samples
    episode_count = 0
    for epoch in range(args.n_epochs):
        t_init = time.time()
        # setup time_tracking
        time_dict = dict(goal_sampler=0,
                         rollout=0,
                         store=0,
                         norm_update=0,
                         policy_train=0,
                         eval=0,
                         epoch=0)

        if rank==0: logger.info('\n\nEpoch #{}'.format(epoch))
        for _ in range(args.n_cycles):
            # sample goal
            t_i = time.time()
            # inits, goals, self_eval = goal_sampler.sample_goal(n_goals=args.num_rollouts_per_mpi, evaluation=False)
            predicates, pairs = goal_sampler.sample_goal(n_goals=args.num_rollouts_per_mpi)
            time_dict['goal_sampler'] += time.time() - t_i

            # collect episodes
            t_i = time.time()

            # Control biased initializations
            if epoch < args.start_biased_init:
                biased_init = False
            else:
                biased_init = args.biased_init
            episodes = rollout_worker.generate_rollout(predicates=predicates, pairs=pairs, true_eval=False, biased_init=biased_init)
            # episodes = rollout_worker.generate_rollout(inits=inits,
            #                                            goals=goals,
            #                                            self_eval=self_eval,
            #                                            true_eval=False,
            #                                            biased_init=biased_init)

            time_dict['rollout'] += time.time() - t_i

            # update goal sampler (add new discovered goals to the list
            # label episodes with the id of the last ag
            # t_i = time.time()
            # episodes = goal_sampler.update(episodes, episode_count)
            # time_dict['gs_update'] += time.time() - t_i

            # store episodes
            t_i = time.time()
            policy.store(episodes)
            time_dict['store'] += time.time() - t_i

            # update normalizer
            t_i = time.time()
            for e in episodes:
                policy._update_normalizer(e)
            time_dict['norm_update'] += time.time() - t_i

            # train policy
            t_i = time.time()
            for _ in range(args.n_batches):
                policy.train()
            time_dict['policy_train'] += time.time() - t_i

            episode_count += args.num_rollouts_per_mpi * args.num_workers

        # t_i = time.time()
        # if goal_sampler.curriculum_learning and rank == 0:
        #     goal_sampler.update_LP()
        # goal_sampler.sync()

        # time_dict['lp_update'] += time.time() - t_i
        time_dict['epoch'] += time.time() - t_init
        time_dict['total'] = time.time() - t_total_init

        if args.evaluations:
            t_i = time.time()
            if rank==0: logger.info('\tRunning eval ..')
            # eval_goals = goal_sampler.valid_goals
            eval_predicates = np.array([0 if i < args.n_test_rollouts//2 else 1 for i in range(args.n_test_rollouts)])
            eval_pairs = np.array([np.random.choice(np.arange(args.env_params['num_blocks']), size=2, replace=False)
                                   for _ in range(args.n_test_rollouts)])
            episodes = rollout_worker.generate_rollout(predicates=eval_predicates, pairs=eval_pairs, true_eval=True, biased_init=False)

            results = np.array([str(e['g'][0][-1]) == str(e['ag'][-1][-1]) for e in episodes]).astype(np.int)
            all_results = MPI.COMM_WORLD.gather(results, root=0)
            time_dict['eval'] += time.time() - t_i
            if rank == 0:
                assert len(all_results) == args.num_workers
                av_res = np.array(all_results).mean(axis=0)
                global_sr = np.mean(av_res)
                log_and_save(logdir, goal_sampler, epoch, episode_count, av_res, global_sr,time_dict)
                if epoch % args.save_freq == 0:
                    policy.save(model_path, epoch)
                    # goal_sampler.save_bucket_contents(bucket_path, epoch)
                if rank==0: logger.info('\tEpoch #{}: SR: {}'.format(epoch, global_sr))


def log_and_save( logdir, goal_sampler, epoch, episode_count, av_res, global_sr, time_dict):
    goal_sampler.save(logdir, epoch, episode_count, av_res, global_sr, time_dict)
    for k, l in goal_sampler.stats.items():
        logger.record_tabular(k, l[-1])
    logger.dump_tabular()



if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # get the params
    args = get_args()

    launch(args)
