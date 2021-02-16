from collections import deque
import numpy as np
from utils import  generate_all_goals_in_goal_space, generate_goals
from mpi4py import MPI
import os
import pickle
import pandas as pd
from mpi_utils import logger


class GoalSampler:
    def __init__(self, args):
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()

        self.goal_dim = args.env_params['goal']

        buckets = generate_goals()
        valid_goals = []
        for k in buckets.keys():
            valid_goals += buckets[k]
        self.valid_goals = np.array(valid_goals)

        self.discovered_goals = []
        self.discovered_goals_str = []

        self.init_stats()

    def sample_goal(self, n_goals, evaluation):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        if evaluation and len(self.discovered_goals) > 0:
            goals = np.random.choice(self.discovered_goals, size=self.num_rollouts_per_mpi)
            self_eval = False
        elif len(self.discovered_goals) == 0:
            # sample randomly in the goal space
            goals = np.random.choice([-1., 1.], size=(n_goals, self.goal_dim))
            self_eval = False
        # if goals have been discovered
        else:
            # sample uniformly from discovered goals
            goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
            goals = np.array(self.discovered_goals)[goal_ids]
            self_eval = False
        return goals, self_eval

    def update(self, episodes, t):
        """
        Update discovered goals list from episodes
        Update list of successes and failures for LP curriculum
        Label each episode with the last ag (for buffer storage)
        """
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

        if self.rank == 0:
            all_episode_list = [e for eps in all_episodes for e in eps]

            for e in all_episode_list:
                if str(e['ag_binary'][-1]) not in self.discovered_goals_str:
                    self.discovered_goals.append(e['ag_binary'][-1].copy())
                    self.discovered_goals_str.append(str(e['ag_binary'][-1]))

        self.sync()

        return episodes

    def sync(self):
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
        self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)

    def build_batch(self, batch_size):
        goal_ids = np.random.choice(self.discovered_goals_oracle_id, size=batch_size)
        return goal_ids

    def init_stats(self):
        self.stats = dict()
        for i in range(self.valid_goals.shape[0]):
            self.stats['Eval_SR_{}'.format(i)] = []  # track the offline success rate of each valid goal
            self.stats['Av_Rew_{}'.format(i)] = []  # track the offline average reward of each valid goal

        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['global_sr'] = []
        self.stats['nb_discovered'] = []
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update',
                'policy_train', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        self.stats['nb_discovered'].append(len(self.discovered_goals))
        for i in range(self.valid_goals.shape[0]):
            self.stats['Eval_SR_{}'.format(i)].append(av_res[i])
            self.stats['Av_Rew_{}'.format(i)].append(av_rew[i])
