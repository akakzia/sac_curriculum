from collections import deque
import numpy as np
from utils import get_idxs_per_relation, get_number_of_floors
from mpi4py import MPI
import os
import pickle
import pandas as pd
from mpi_utils import logger

EXPLORATION_EPS = 12000


class GoalSampler:
    def __init__(self, args):
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.use_masks = args.masks
        self.mask_application = args.mask_application

        self.goal_dim = args.env_params['goal']
        self.relation_ids = get_idxs_per_relation(n=args.n_blocks)

        self.discovered_goals = []
        self.discovered_goals_str = []

        self.curriculum_learning = args.curriculum_learning
        self.queue_length = args.queue_length
        self.epsilon = args.epsilon_curr
        self.n_blocks = args.n_blocks

        if self.curriculum_learning:
            # initialize deques of successes and failures for all goals
            self.successes_and_failures = [[] for _ in range(self.n_blocks)]
            # fifth bucket contains not discovered goals
            self.LP = np.zeros([self.n_blocks])
            self.C = np.zeros([self.n_blocks])
            self.p = np.ones([self.n_blocks]) / self.n_blocks

            self.buckets = dict(zip(range(self.n_blocks), [[] for _ in range(self.n_blocks)]))
            self.active_buckets = np.zeros(self.n_blocks)

        self.init_stats()

    def sample_masks(self, n):
        """Samples n masks uniformly"""
        if not self.use_masks:
            # No masks
            return np.zeros((n, self.goal_dim))
        masks = np.zeros((n, self.goal_dim))
        # Select number of masks to apply per goal
        n_masks = np.random.randint(self.relation_ids.shape[0], size=n)
        # Get idxs to be masked
        relations_to_mask = [np.random.choice(np.arange(self.relation_ids.shape[0]), size=i, replace=False) for i in n_masks]
        re = [np.concatenate(self.relation_ids[r]) if self.relation_ids[r].shape[0] > 0 else None for r in relations_to_mask]
        # apply masks
        for mask, ids_to_mask in zip(masks, re):
            if ids_to_mask is not None:
                mask[ids_to_mask] = 1
        return masks

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
            self_eval = False
            if len(self.discovered_goals) == 0:
                goals = np.random.choice([-1., 1.], size=(n_goals, self.goal_dim))
                masks = np.zeros((n_goals, self.goal_dim))
                # masks = np.random.choice([0., 1.], size=(n_goals, self.goal_dim))
            # if no curriculum learning
            elif self.curriculum_learning:
                cond = np.array([len(self.buckets[i]) > 0 for i in range(self.n_blocks)]).all()
                self_eval = True if np.random.random() < 0.1 else False
                if cond:
                    # if self-evaluation then sample randomly from discovered goals
                    if self_eval:
                        buckets = np.random.choice(range(self.n_blocks), size=n_goals)
                    else:
                        buckets = np.random.choice(range(self.n_blocks), p=self.p, size=n_goals)
                    goals = []
                    for i_b, b in enumerate(buckets):
                        goals.append(self.buckets[b][np.random.choice(np.arange(len(self.buckets[b])))])
                    goals = np.array(goals)
                    masks = self.sample_masks(n_goals)
                else:
                    # sample uniformly from discovered goals
                    goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
                    goals = np.array(self.discovered_goals)[goal_ids]
                    masks = self.sample_masks(n_goals)
                    # masks = np.array(self.masks_list)[np.random.choice(range(self.n_masks), size=n_goals)]
                    # self_eval = False
            else:
                # sample uniformly from discovered goals
                goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
                goals = np.array(self.discovered_goals)[goal_ids]
                masks = self.sample_masks(n_goals)
        return goals, masks, self_eval

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
                # Add last achieved goal to memory if first time encountered
                if str(e['ag_binary'][-1]) not in self.discovered_goals_str:
                    self.discovered_goals.append(e['ag_binary'][-1].copy())
                    self.discovered_goals_str.append(str(e['ag_binary'][-1]))
                    if self.curriculum_learning:
                        nb_floors = get_number_of_floors(e['ag_binary'][-1], self.n_blocks)
                        self.buckets[nb_floors].append(e['ag_binary'][-1].copy())

                if self.curriculum_learning:
                    # put achieved goals in buckets according to the number of floors
                    nb_floors = get_number_of_floors(e['ag_binary'][-1], self.n_blocks)
                    self.active_buckets[nb_floors] = 1.
                    if e['self_eval']:
                        self.successes_and_failures[nb_floors].append(e['success'][-1].astype(np.float))
                        if len(self.successes_and_failures[nb_floors]) > self.queue_length:
                            self.successes_and_failures[nb_floors] = self.successes_and_failures[nb_floors][-self.queue_length:]

        self.sync()

        # Apply masks
        if self.use_masks:
            for e in episodes:
                if self.mask_application == 'hindsight':
                    e['g'] = e['g'] * (1 - e['masks'][0]) + e['ag'][:-1] * e['masks'][0]
                elif self.mask_application == 'initial':
                    e['g'] = e['g'] * (1 - e['masks'][0]) + e['ag'][0] * e['masks'][0]
                elif self.mask_application == 'opaque':
                    e['g'] = e['g'] * (1 - e['masks'][0]) - 10 * e['masks'][0]
                else:
                    raise NotImplementedError

        bs = []
        if self.curriculum_learning:
            if t > EXPLORATION_EPS:
                self.update_LP()
            for e in episodes:
                nb_floors = get_number_of_floors(e['ag_binary'][-1], self.n_blocks)
                bs.append(nb_floors)
                self.active_buckets[nb_floors] = 1.

        return episodes, bs

    def update_LP(self):
        # compute C, LP per bucket
        for k in self.buckets.keys():
            n_points = len(self.successes_and_failures[k])
            if n_points > 100:
                sf = np.array(self.successes_and_failures[k])
                self.C[k] = np.mean(sf[n_points // 2:])
                # self.LP[k] = np.abs(np.sum(sf[n_points // 2:, 1]) - np.sum(sf[: n_points // 2, 1])) / n_points
                self.LP[k] = np.abs(np.mean(sf[n_points // 2:]) - np.mean(sf[: n_points // 2]))

        # compute p
        if self.LP.sum() == 0:
            self.p = np.ones([self.n_blocks]) / self.n_blocks
        else:
            self.p = self.LP / self.LP.sum()
            # self.p = self.epsilon * (1 - self.C) / (1 - self.C).sum() + (1 - self.epsilon) * self.LP / self.LP.sum()

        if self.p.sum() > 1:
            self.p[np.argmax(self.p)] -= self.p.sum() - 1
        elif self.p.sum() < 1:
            self.p[-1] = 1 - self.p[:-1].sum()

    def generate_eval_goals(self):
        """ Generates a set of goals for evaluation. This set comprises :
        - One relation with close == True .
        - One relation with above == True
        - Two relations with close == True in one of them
        - Two relations with close == True in both of them
        - Two relations with above == True in one and close == False in the other
        - Two relations with above == True in one and close == True in the other
        - Two relations with above == True in one and above == True in the other
        - Three whole relations for the 7 above cases"""
        if self.use_masks:
            masks = np.array([np.array([0, 1, 1, 0, 1, 0, 1, 1, 1]), np.array([0, 1, 1, 0, 1, 0, 1, 1, 1]),
                              np.array([0, 0, 1, 0, 0, 0, 1, 0, 1]), np.array([0, 0, 1, 0, 0, 0, 1, 0, 1]),
                              np.array([0, 0, 1, 0, 0, 0, 1, 0, 1]), np.array([0, 0, 1, 0, 0, 0, 1, 0, 1]),
                              np.array([0, 1, 0, 0, 1, 0, 0, 1, 0]),
                              np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(9)])
        else:
            masks = np.zeros((12, 9))
        gs = np.array([np.array([1., -10., -10., -1., -10., -1., -10., -10., -10.]), np.array([1., -10., -10., 1., -10., -1., -10., -10., -10.]),

                       np.array([1., -1., -10., -1., -1., -1., -10., -1., -10.]), np.array([1., 1., -10., -1., -1., -1., -10., -1., -10.]),
                       np.array([1., -1., -10., -1., -1., 1., -10., -1., -10.]), np.array([1., 1., -10., -1., 1., -1., -10., -1., -10.]),
                       np.array([1., -10., 1., 1., -10., -1., 1., -10., -1.]),

                       np.array([1., -1., -1., -1., -1., -1., -1., -1., -1.]), np.array([1., -1., -1., 1., -1., -1., -1., -1., -1.]),

                       np.array([1., 1., -1., -1., -1., -1., -1., -1., -1.]),
                       np.array([1., 1., 1., -1., -1., 1., -1., -1., -1.]),
                       np.array([1., -1., 1., 1., -1., -1., 1., -1., -1.])
                       ])
        return gs, masks

    def sync(self):
        if self.curriculum_learning:
            self.p = MPI.COMM_WORLD.bcast(self.p, root=0)
            self.LP = MPI.COMM_WORLD.bcast(self.LP, root=0)
            self.C = MPI.COMM_WORLD.bcast(self.C, root=0)
            self.buckets = MPI.COMM_WORLD.bcast(self.buckets, root=0)
            self.successes_and_failures = MPI.COMM_WORLD.bcast(self.successes_and_failures, root=0)
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
        self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)

    def build_batch(self, batch_size):
        # only consider buckets filled with discovered goals
        LP = self.LP
        # C = self.C
        if LP.sum() == 0:
            p = np.ones([self.n_blocks]) * self.active_buckets / np.count_nonzero(self.active_buckets)
        else:
            p = self.epsilon * np.ones([self.n_blocks]) * self.active_buckets / self.n_blocks + (1 - self.epsilon) * LP / LP.sum()
            # p = LP / LP.sum()
            # p = self.epsilon * (1 - C) / (1 - C).sum() + (1 - self.epsilon) * LP / LP.sum()
        if p.sum() > 1:
            p[np.argmax(self.p)] -= p.sum() - 1
        elif p.sum() < 1:
            if np.count_nonzero(self.active_buckets) == self.n_blocks:
                p[-1] = 1 - p[:-1].sum()
            else:
                p[0] = p[0] + 1 - p.sum()
        buckets = np.random.choice(range(self.n_blocks), p=p, size=batch_size)
        # buckets = np.random.choice(range(self.num_buckets), p=p) * np.ones(batch_size)
        # goal_ids = []
        # for b in buckets:
        #     if len(self.buckets[b]) > 0:
        #         goal_ids.append(np.random.choice(self.buckets[b]))
        #     else:
        #         goal_ids.append(-1)  # this will lead the buffer to sample a random episode
        # assert len(goal_ids) == batch_size
        return buckets

    def init_stats(self):
        self.stats = dict()
        # Number of classes of eval
        if self.goal_dim == 30:
            n = 12
        else:
            n = 6
        for i in np.arange(1, n+1):
            self.stats['Eval_SR_{}'.format(i)] = []
            self.stats['Av_Rew_{}'.format(i)] = []

        if self.curriculum_learning:
            for i in range(self.n_blocks):
                self.stats['B_{}_LP'.format(i)] = []
                self.stats['B_{}_C'.format(i)] = []
                self.stats['B_{}_p'.format(i)] = []

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
        for g_id in np.arange(1, len(av_res) + 1):
            self.stats['Eval_SR_{}'.format(g_id)].append(av_res[g_id-1])
            self.stats['Av_Rew_{}'.format(g_id)].append(av_rew[g_id-1])
            # self.stats['#Rew_{}'.format(g_id)].append(self.rew_counters[oracle_id])
            # self.stats['#Target_{}'.format(g_id)].append(self.target_counters[oracle_id])
        if self.curriculum_learning:
            for i in range(self.n_blocks):
                self.stats['B_{}_LP'.format(i)].append(self.LP[i])
                self.stats['B_{}_C'.format(i)].append(self.C[i])
                self.stats['B_{}_p'.format(i)].append(self.p[i])
