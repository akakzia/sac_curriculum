import numpy as np
from mpi4py import MPI
import env
import gym
import os
from arguments import get_args
from rl_modules.sac_agent import SACAgent
import random
import torch
from rollout import RolloutWorker
from goal_sampler import GoalSampler
from utils import init_storage, get_instruction2
import time
from mpi_utils import logger
from language.build_dataset import sentence_from_configuration, NO_SYNONYMS

def get_env_params(env):
    obs = env.reset()

    # close the environment
    params = {'obs': obs['observation'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

def launch(args):

    rank = MPI.COMM_WORLD.Get_rank()

    t_total_init = time.time()

    # Make the environment
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
        logdir, model_path, bucket_path = init_storage(args)
        logger.configure(dir=logdir)
        logger.info(vars(args))

    args.env_params = get_env_params(env)

    language_goals = get_instruction2()
    print('List of instructions ({}):'.format(len(language_goals)), language_goals)

    # Initialize RL Agent
    policy = SACAgent(language_goals, args, env.compute_reward)

    # Initialize Rollout Worker
    rollout_worker = RolloutWorker(env, policy, args)

    logclass = Logger(language_goals)
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
                         lp_update=0,
                         eval=0,
                         epoch=0)

        # log current epoch
        if rank == 0: logger.info('\n\nEpoch #{}'.format(epoch))

        # Cycles loop
        for _ in range(args.n_cycles):

            # Sample goals
            t_i = time.time()
            language_goal_ep = np.random.choice(language_goals, size=args.num_rollouts_per_mpi)
            self_eval = False
            time_dict['goal_sampler'] += time.time() - t_i

            # Control biased initializations
            if epoch < args.start_biased_init:
                biased_init = False
            else:
                biased_init = args.biased_init

            # Environment interactions
            t_i = time.time()
            episodes = rollout_worker.generate_rollout(true_eval=False,  # these are not offline evaluation episodes
                                                       biased_init=biased_init,
                                                       language_goal=language_goal_ep)  # whether initializations should be biased.
            time_dict['rollout'] += time.time() - t_i

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

        time_dict['epoch'] += time.time() -t_init
        time_dict['total'] = time.time() - t_total_init

        if args.evaluations:
            if rank==0: logger.info('\tRunning eval ..')
            # Performing evaluations
            t_i = time.time()

            episodes = rollout_worker.generate_rollout(true_eval=True,  # this is offline evaluations
                                                       biased_init=False,
                                                       language_goal=language_goals)

            # Extract the results
            results = np.array([e['language_goal'] in sentence_from_configuration(config=e['ag'][-1], all=True) for e in episodes]).astype(np.int)

            all_results = MPI.COMM_WORLD.gather(results, root=0)
            time_dict['eval'] += time.time() - t_i

            # Logs
            if rank == 0:
                assert len(all_results) == args.num_workers  # MPI test
                av_res = np.array(all_results).mean(axis=0)
                global_sr = np.mean(av_res)
                logclass.save(epoch, episode_count, av_res, global_sr, time_dict)
                for k, l in logclass.stats.items():
                    logger.record_tabular(k, l[-1])
                logger.dump_tabular()

                # Saving policy models
                if epoch % args.save_freq == 0:
                    policy.save(model_path, epoch)
                if rank==0: logger.info('\tEpoch #{}: SR: {}'.format(epoch, global_sr))


class Logger():
    def __init__(self, language_goals):
        self.stats = dict()
        self.nb_goals = len(language_goals)
        for i in range(self.nb_goals):
            self.stats['Eval_SR_{}'.format(i)] = []  # track the offline success rate of each valid goal

        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['global_sr'] = []
        # Track the time spent in each function
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update',
                  'policy_train', 'lp_update', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, global_sr, time_dict):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        for g_id in range(self.nb_goals):
            self.stats['Eval_SR_{}'.format(g_id)].append(av_res[g_id])





if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # Get parameters
    args = get_args()

    launch(args)
