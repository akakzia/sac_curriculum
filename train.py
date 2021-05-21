import numpy as np
from mpi4py import MPI
import env
import gym
import os
from arguments import get_args
from rl_modules.rl_agent import RLAgent
import random
import torch
from rollout import RolloutWorker
from goal_sampler import GoalSampler
from utils import init_storage, get_eval_goals
import time
from mpi_utils import logger
from graph.semantic_graph import SemanticGraph
from graph.agent_network import AgentNetwork


def get_env_params(env):
    obs = env.reset()

    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

def launch(args):

    rank = MPI.COMM_WORLD.Get_rank()

    t_total_init = time.time()

    # Make the environment
    args.env_name = 'FetchManipulate{}Objects-v0'.format(args.n_blocks)
    env = gym.make(args.env_name)

    # set random seeds for reproducibility
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # get saving paths
    if rank == 0:
        logdir, model_path = init_storage(args)
        logger.configure(dir=logdir)
        logger.info(vars(args))

    args.env_params = get_env_params(env)

    goal_sampler = GoalSampler(args)

    # Initialize RL Agent
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward, goal_sampler)
    else:
        raise NotImplementedError

    # Initialize Rollout Worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    # initialize graph components : 
    semantic_graph = SemanticGraph.load(args.n_blocks)
    agent_network = AgentNetwork(semantic_graph,args)

    # Main interaction loop
    episode_count = 0
    for epoch in range(args.n_epochs):
        t_init = time.time()

        # setup time_tracking
        time_dict = dict(goal_sampler=0,
                         rollout=0,
                         gs_update=0,
                         store=0,
                         norm_update=0,
                         policy_train=0,
                         eval=0,
                         epoch=0)

        # log current epoch
        if rank == 0: logger.info('\n\nEpoch #{}'.format(epoch))

        # Cycles loop
        for _ in range(args.n_cycles):
            # Sample goals
            t_i = time.time()
            goals = agent_network.sample_goal(args.num_rollouts_per_mpi*args.n_blocks)
            time_dict['goal_sampler'] += time.time() - t_i

            # Control biased initializations
            if epoch < args.start_biased_init:
                biased_init = False
            else:
                biased_init = args.biased_init

            # Environment interactions
            t_i = time.time()
            episodes = rollout_worker.guided_rollouts(goals=goals,  # list of goal configurations
                                                    self_eval=False,  # whether the agent performs self-evaluations
                                                    true_eval=False,  # these are not offline evaluation episodes
                                                    semantic_graph = semantic_graph, 
                                                    max_episodes=args.num_rollouts_per_mpi*args.n_blocks,
                                                    episode_duration=args.episode_duration,
                                                    biased_init=biased_init,  # whether initializations should be biased.
                                                    )
            time_dict['rollout'] += time.time() - t_i

            # Goal Sampler updates
            t_i = time.time()
            episodes = goal_sampler.update(episodes, episode_count)
            time_dict['gs_update'] += time.time() - t_i

            # Storing episodes
            t_i = time.time()
            policy.store(episodes)
            time_dict['store'] += time.time() - t_i

            # Updating observation normalization
            t_i = time.time()
            for e in episodes:
                policy._update_normalizer(e)
            time_dict['norm_update'] += time.time() - t_i

            # Policy updates
            t_i = time.time()
            for _ in range(args.n_batches):
                policy.train()
            time_dict['policy_train'] += time.time() - t_i
            episode_count += args.num_rollouts_per_mpi * args.num_workers
            # Agent Network Update : 
            agent_network.update(episodes)

        time_dict['epoch'] += time.time() -t_init
        time_dict['total'] = time.time() - t_total_init

        if args.evaluations:
            if rank==0: logger.info('\tRunning eval ..')
            # Performing evaluations
            t_i = time.time()
            eval_goals = []
            if args.n_blocks == 3:
                instructions = ['close_1', 'close_2', 'close_3', 'stack_2', 'pyramid_3', 'stack_3']
            elif args.n_blocks == 5:
                instructions = ['close_1', 'close_2', 'close_3', 'stack_2', 'stack_3', '2stacks_2_2', '2stacks_2_3', 'pyramid_3',
                                'mixed_2_3', 'trapeze_2_3', 'stack_4', 'stack_5']
            else:
                raise NotImplementedError
            for instruction in instructions:
                eval_goal = get_eval_goals(instruction, n=args.n_blocks)
                eval_goals.append(eval_goal.squeeze(0))
            eval_goals = np.array(eval_goals)
            eval_masks = np.array(np.zeros((eval_goals.shape[0], args.n_blocks * (args.n_blocks - 1) * 3 // 2)))
            
            episodes = rollout_worker.guided_rollouts(goals=eval_goals,  # list of goal configurations
                                                    self_eval=True,  # whether the agent performs self-evaluations
                                                    true_eval=True,  # these are not offline evaluation episodes
                                                    semantic_graph = semantic_graph, 
                                                    episode_duration=args.episode_duration,
                                                    max_episodes=None,
                                                    biased_init=False,  # whether initializations should be biased.
                                                    )

            results = np.array([e['success'][-1].astype(np.float32) for e in episodes])
            rewards = np.array([e['rewards'][-1] for e in episodes])
            all_results = MPI.COMM_WORLD.gather(results, root=0)
            all_rewards = MPI.COMM_WORLD.gather(rewards, root=0)
            time_dict['eval'] += time.time() - t_i

            # Logs
            if rank == 0:
                assert len(all_results) == args.num_workers  # MPI test
                av_res = np.array(all_results).mean(axis=0)
                av_rewards = np.array(all_rewards).mean(axis=0)
                global_sr = np.mean(av_res)
                
                agent_network.log(logger)
                logger.record_tabular('replay_nb_edges', policy.buffer.get_nb_edges())
                log_and_save(goal_sampler, epoch, episode_count, av_res, av_rewards, global_sr, time_dict)

                # Saving policy models
                if epoch % args.save_freq == 0:
                    policy.save(model_path, epoch)
                if rank==0: logger.info('\tEpoch #{}: SR: {}'.format(epoch, global_sr))


def log_and_save( goal_sampler, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
    goal_sampler.save(epoch, episode_count, av_res, av_rew, global_sr, time_dict)
    for k, l in goal_sampler.stats.items():
        logger.record_tabular(k, l[-1])
    logger.dump_tabular()


if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # Get parameters
    args = get_args()

    launch(args)
