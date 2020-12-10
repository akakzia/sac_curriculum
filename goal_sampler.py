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

        self.goal_dim = 9

        self.discovered_goals = []
        self.discovered_goals_str = []

        self.init_stats()

    @staticmethod
    def apply_constraints(constraints, eval=False):
        """ gs and constraints have the same shape across first dimensions
        constraints values are comprised between 1 and second shape of gs
        Given an array of goals and an array of constraints
        Returns an array of partial goals"""
        # assert gs.shape[0] == constraints.shape[0]
        # assert np.max(constraints) <= gs.shape[1]
        # for g, c in zip(gs, constraints):
        #     ids_masks = np.random.choice(range(gs.shape[1]), size=gs.shape[1] - c, replace=False)
        #     g[ids_masks] = 0.
        if not eval:
            gs = []
            p = [2/9, 2/9, 2/9, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18]
            for c in constraints:
                g = np.zeros(9)
                ids_constraints = np.random.choice(range(9), size=c, p=p)
                g[ids_constraints] = 1.
                gs.append(g)
            gs = np.array(gs)
        else:
            c = constraints[0]
            g1 = np.zeros(9)
            g2 = np.zeros(9)
            id_close = np.random.randint(0, 3, size=c)
            id_above = np.random.randint(3, 9, size=c)
            g1[id_close] = 1.
            g2[id_above] = 1.
            gs = np.array([g1, g2])
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
            # if len(self.discovered_goals) == 0:
            #     goals = np.random.choice([1., -1.], size=(n_goals, self.goal_dim))
            #     self_eval = False
            # # if no curriculum learning
            # else:
            #     # sample uniformly from discovered goals
            #     goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
            #     num_constraints = np.random.randint(1, self.goal_dim+1, size=n_goals)
            #     goals = np.array(self.discovered_goals)[goal_ids]
            #     goals = self.apply_constraints(goals, num_constraints)
            #     self_eval = False
            num_constraints = np.random.randint(1, 2, size=n_goals)
            goals = self.apply_constraints(num_constraints)
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

    def sync(self):
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
        self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)

    def build_batch(self, batch_size):
        goal_ids = np.random.choice(np.arange(len(self.discovered_goals)), size=batch_size)
        return goal_ids

    def init_stats(self):
        self.stats = dict()
        self.stats['Eval_SR_1_close'] = []
        self.stats['Eval_SR_1_above'] = []
        # for i in np.arange(1, self.goal_dim+1):
        #     self.stats['Eval_SR_{}'.format(i)] = []
        #     self.stats['Av_Rew_{}'.format(i)] = []
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
        self.stats['Eval_SR_1_close'].append(av_res[0])
        self.stats['Eval_SR_1_above'].append(av_res[1])
        # for g_id in np.arange(1, self.goal_dim+1):
        #     self.stats['Eval_SR_{}'.format(g_id)].append(av_res[g_id-1])
        #     self.stats['Av_Rew_{}'.format(g_id)].append(av_rew[g_id-1])
            # self.stats['#Rew_{}'.format(g_id)].append(self.rew_counters[oracle_id])
            # self.stats['#Target_{}'.format(g_id)].append(self.target_counters[oracle_id])
