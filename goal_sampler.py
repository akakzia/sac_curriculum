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
        self.continuous = args.algo == 'continuous'
        self.curriculum_learning = args.curriculum_learning
        self.automatic_buckets = args.automatic_buckets
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()

        self.goal_dim = args.env_params['goal']

        self.discovered_goals = []
        self.discovered_goals_str = []

        self.init_stats()

    @staticmethod
    def apply_constraints(gs):
        """ gs and constraints have the same shape across first dimensions
        constraints values are comprised between 1 and second shape of gs
        Given an array of goals and an array of constraints
        Returns an array of partial goals"""
        # DEBUG 3 constraints for the pairwise predicates
        goal_ids = [[2, 8, 14, 20, 26, 3, 9, 15, 21, 27, 4, 10, 16, 22, 28, 5, 11, 17, 23, 29],
                    [0, 6, 12, 18, 24, 1, 7, 13, 19, 25, 4, 10, 16, 22, 28, 5, 11, 17, 23, 29],
                    [0, 6, 12, 18, 24, 1, 7, 13, 19, 25, 2, 8, 14, 20, 26, 3, 9, 15, 21, 27],
                    [4, 10, 16, 22, 28, 5, 11, 17, 23, 29],
                    [2, 8, 14, 20, 26, 3, 9, 15, 21, 27],
                    [0, 6, 12, 18, 24, 1, 7, 13, 19, 25],
                    [],
                    [],
                    []]
        for g in gs:
            ids_masks = np.random.randint(0, len(goal_ids))
            g[goal_ids[ids_masks]] = 0.
        return gs

    def sample_goal(self, n_goals, evaluation):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        if evaluation and len(self.discovered_goals) > 0:
            goals = np.random.choice(self.discovered_goals, size=self.num_rollouts_per_mpi)
            self_eval = False
        else:
            if len(self.discovered_goals) == 0:
                goals = np.zeros((n_goals, self.goal_dim))
                ids = np.random.choice(np.arange(self.goal_dim), size=(n_goals, 3))
                for i in range(n_goals):
                    goals[i, ids[i]] = -1.
                # goals = np.random.choice([1., -1.], size=(n_goals, self.goal_dim))
                self_eval = False
            # if no curriculum learning
            else:
                # sample uniformly from discovered goals
                goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
                # num_constraints = np.random.randint(1, self.goal_dim+1, size=n_goals)
                goals = np.array(self.discovered_goals)[goal_ids]
                goals = self.apply_constraints(goals)
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
            all_episode_list = []
            for eps in all_episodes:
                all_episode_list += eps

            # for e in all_episode_list:
            #     reached_oracle_id = self.g_str_to_oracle_id[str(e['ag_binary'][-1])]
            #     target_oracle_id = self.g_str_to_oracle_id[str(e['g_binary'][0])]
            #     self.rew_counters[reached_oracle_id] += 1
            #     self.target_counters[target_oracle_id] += 1
            # find out if new goals were discovered
            # label each episode with the oracle id of the last ag (to know where to store it in buffers)
            if not self.curriculum_learning or self.automatic_buckets:
                new_goal_found = False
                for e in all_episode_list:
                    if str(e['ag_binary'][-1]) not in self.discovered_goals_str:
                        new_goal_found = True
                        self.discovered_goals.append(e['ag_binary'][-1].copy())
                        self.discovered_goals_str.append(str(e['ag_binary'][-1]))

        self.sync()

        return episodes

    @staticmethod
    def generate_eval_goals():
        """ Generates a set of goals for evaluation. This set comprises :
        - One relation with close == True .
        - One relation with above == True
        - Two relations with close == True in one of them
        - Two relations with close == True in both of them
        - Two relations with above == True in one and close == False in the other
        - Two relations with above == True in one and close == True in the other
        - Two relations with above == True in one and above == True in the other
        - Three whole relations for the 7 above cases"""
        anchor = np.zeros(30)
        left = np.array([1., -1., -1., -1., -1, -1., -1., -1., -1., -1.])
        above = np.array([-1., -1., -1., -1., -1, -1., -1., -1., 1., -1.])
        far = np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])
        g_1_relation_close = anchor.copy()
        g_2_relations_close = anchor.copy()
        g_3_relations_close = anchor.copy()
        g_1_relation_above = anchor.copy()
        g_2_relations_above = anchor.copy()
        g_3_relations_above = anchor.copy()
        ids_0_1 = [0, 1, 6, 7, 12, 13, 18, 19, 24, 25]
        ids_0_2 = [2, 3, 8, 9, 14, 15, 20, 21, 26, 27]
        ids_1_2 = [4, 5, 10, 11, 16, 17, 22, 23, 28, 29]
        g_1_relation_close[ids_0_1] = left
        g_1_relation_above[ids_0_1] = above

        g_2_relations_close[ids_0_1] = left
        g_2_relations_close[ids_0_2] = far

        g_2_relations_above[ids_0_1] = above
        g_2_relations_above[ids_0_2] = far

        g_3_relations_close[ids_0_1] = left
        g_3_relations_close[ids_0_2] = far
        g_3_relations_close[ids_1_2] = far

        g_3_relations_above[ids_0_1] = above
        g_3_relations_above[ids_0_2] = far
        g_3_relations_above[ids_1_2] = far

        return np.array([g_1_relation_close, g_1_relation_above, g_2_relations_close, g_2_relations_above, g_3_relations_close, g_3_relations_above])
        # return np.array([np.array([1., 0., 0., -1., -1., 0., 0., 0., 0.]), np.array([1., 0., 0., 1., -1., 0., 0., 0., 0.]),
        #
        #                  np.array([1., -1., 0., -1., -1., -1., -1., 0., 0.]), np.array([1., 1., 0., -1., -1., -1., -1., 0., 0.]),
        #                  np.array([1., -1., 0., -1., 1., -1., -1., 0., 0.]), np.array([1., 1., 0., -1., 1., -1., -1., 0., 0.]),
        #                  np.array([1., 0., 1., 1., -1., 0., 0., 1., -1.]),
        #
        #                  np.array([1., -1., -1., -1., -1., -1., -1., -1., -1.]), np.array([1., -1., -1., 1., -1., -1., -1., -1., -1.]),
        #
        #                  np.array([1., 1., -1., -1., -1., -1., -1., -1., -1.]),
        #                  np.array([1., 1., 1., -1., 1., -1., -1., -1., -1.]),
        #                  np.array([1., -1., 1., 1., -1., -1., -1., 1., -1.])
        #                  ])

    def sync(self):
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
        self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)

    def build_batch(self, batch_size):
        goal_ids = np.random.choice(np.arange(len(self.discovered_goals)), size=batch_size)
        return goal_ids

    def init_stats(self):
        self.stats = dict()
        for i in np.arange(6):
            self.stats['Eval_SR_{}'.format(i)] = []
            self.stats['Av_Rew_{}'.format(i)] = []
        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['global_sr'] = []
        self.stats['nb_discovered'] = []
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update',
                  'policy_train', 'lp_update', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        self.stats['nb_discovered'].append(len(self.discovered_goals))
        for g_id in np.arange(6):
            self.stats['Eval_SR_{}'.format(g_id)].append(av_res[g_id])
            self.stats['Av_Rew_{}'.format(g_id)].append(av_rew[g_id])
            # self.stats['#Rew_{}'.format(g_id)].append(self.rew_counters[oracle_id])
            # self.stats['#Target_{}'.format(g_id)].append(self.target_counters[oracle_id])
