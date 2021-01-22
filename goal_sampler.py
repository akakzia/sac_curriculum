from collections import deque
import numpy as np
from utils import  generate_all_goals_in_goal_space, generate_goals
from mpi4py import MPI
import os
import pickle
import pandas as pd
from mpi_utils import logger
from itertools import combinations, permutations
from utils import get_eval_goals, get_eval_masks


class GoalSampler:
    def __init__(self, args):
        self.continuous = args.algo == 'continuous'
        self.curriculum_learning = args.curriculum_learning
        self.automatic_buckets = args.automatic_buckets
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()

        self.goal_dim = args.env_params['goal']

        self.masks = self._get_masks(args.env_params['num_objects'])

        self.n_masks = self.masks.shape[0]

        self.discovered_goals = []
        self.discovered_goals_str = []

        self.init_stats()

    def _get_masks(self, n):
        """
        Creates a set of masks
        n : number of objects
        """
        configuration_mapping = list(combinations(np.arange(n), 2)) + list(permutations(np.arange(n), 2))
        pairs_list = list(combinations(np.arange(n), 2))
        atomic_masks = []
        for pair in pairs_list:
            ids = [i for i, k in enumerate(configuration_mapping) if set(k) == set(pair)]
            temp = np.ones(self.goal_dim)
            temp[ids] = 0.
            atomic_masks.append(temp)

        res = atomic_masks

        all_comb = [list(combinations(atomic_masks, i)) for i in range(2, 10)]

        for comb_list in all_comb:
            for comb in comb_list:
                temp = 1 - sum([1 - m for m in comb])
                res.append(temp)

        return np.array(res)

    def sample_goal(self, n_goals, evaluation):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        if evaluation and len(self.discovered_goals) > 0:
            goals = np.random.choice(self.discovered_goals, size=self.num_rollouts_per_mpi)
            masks = np.zeros((n_goals, self.goal_dim))
            self_eval = False
        else:
            # if len(self.discovered_goals) == 0:
            #     goals = np.random.choice([-1., 1.], size=(n_goals, self.goal_dim))
            #     # masks = np.zeros((n_goals, self.goal_dim))
            #     masks = np.random.choice([0., 1.], size=(n_goals, self.goal_dim))
            #     self_eval = False
            # # if no curriculum learning
            # else:
                # sample uniformly from discovered goals
                # goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
                # goals = np.array(self.discovered_goals)[goal_ids]
                # masks = self.masks[np.random.choice(range(self.n_masks), size=n_goals)]
            instructions = ['close_1', 'close_3', 'pyramid_3', 'stack_5']
            goals = []
            masks = []
            for _ in range(n_goals):
                instruction = np.random.choice(instructions)
                goal = get_eval_goals(instruction)
                gs, ms = get_eval_masks(goal)
                goals.append(gs[-1])
                masks.append(ms[-1])
            goals = np.array(goals)
            masks = np.array(masks)
            self_eval = False
        return goals, masks, self_eval

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

    def init_stats(self):
        self.stats = dict()
        for i in range(20):
            self.stats['Eval_SR_{}'.format(i)] = []
            self.stats['Av_R_{}'.format(i)] = []
        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['average_reward'] = []
        self.stats['global_sr'] = []
        self.stats['nb_discovered'] = []
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update',
                  'policy_train', 'lp_update', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, av_rewards, avg_reward, global_sr, time_dict):
        for i in range(len(av_res)):
            self.stats['Eval_SR_{}'.format(i)].append(av_res[i])
            self.stats['Av_R_{}'.format(i)].append(av_rewards[i])
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        self.stats['average_reward'].append(avg_reward)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        self.stats['nb_discovered'].append(len(self.discovered_goals))
