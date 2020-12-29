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
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()

        self.goal_dim = args.env_params['goal']

        self.relation_to_ids = args.env_params['relation_to_ids']

        self.n_relations = len(self.relation_to_ids)

        self.queue_length = 2 * args.queue_length
        self.epsilon = args.curriculum_eps

        if self.curriculum_learning:
            # initialize deques of successes and failures for all goals
            self.successes_and_failures = [deque(maxlen=self.queue_length) for _ in range(self.n_relations)]
            # fifth bucket contains not discovered goals
            self.LP = np.zeros([self.n_relations])
            self.C = np.zeros([self.n_relations])
            self.p = np.ones([self.n_relations]) / self.n_relations

        self.discovered_goals = []
        self.discovered_goals_str = []

        self.init_stats()

    def apply_constraints(self, gs):
        """ gs and constraints have the same shape across first dimensions
        constraints values are comprised between 1 and second shape of gs
        Given an array of goals and an array of constraints
        Returns an array of partial goals"""
        if not self.curriculum_learning:
            self_eval = False
            for g in gs:
                n_masked_relations = np.random.randint(0, self.n_relations)
                masked_pairs = np.random.choice(list(self.relation_to_ids.keys()), size=n_masked_relations, replace=False)
                for p in masked_pairs:
                    g[self.relation_to_ids[p]] = 0.
        else:
            # decide whether to self evaluate
            self_eval = True if np.random.random() < 0.1 else False
            for g in gs:
                if self_eval:
                    n_masked_relations = np.random.randint(0, self.n_relations)
                else:
                    n_masked_relations = np.random.choice(np.arange(self.n_relations), p=self.p)
                masked_pairs = np.random.choice(list(self.relation_to_ids.keys()), size=n_masked_relations, replace=False)
                for p in masked_pairs:
                    g[self.relation_to_ids[p]] = 0.
        return gs, self_eval

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
                self_eval = False
            # if no curriculum learning
            else:
                # sample uniformly from discovered goals
                goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
                goals = np.array(self.discovered_goals)[goal_ids].copy()
                goals, self_eval = self.apply_constraints(goals)
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

            for e in all_episode_list:
                n_masked_relations = (self.goal_dim - np.count_nonzero(e['g'][0])) // 3
                if str(e['ag_binary'][-1]) not in self.discovered_goals_str:
                    self.discovered_goals.append(e['ag_binary'][-1].copy())
                    self.discovered_goals_str.append(str(e['ag_binary'][-1]))
                if e['self_eval']:
                    success = e['success'][-1]
                    try:
                        self.successes_and_failures[n_masked_relations].append(success)
                    except:
                        pass

        self.sync()
        for e in episodes:
            n_masked_relations = (self.goal_dim - np.count_nonzero(e['g'][0])) // 3
            e['n_masked_relations'] = n_masked_relations

        return episodes

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
        res = []
        for r in range(self.n_relations):
            g_id = np.random.choice(np.arange(len(self.discovered_goals)))
            g = self.discovered_goals[g_id].copy()
            masked_pairs = np.random.choice(list(self.relation_to_ids.keys()), size=r, replace=False)
            for p in masked_pairs:
                g[self.relation_to_ids[p]] = 0.
            res.append(g.copy())
        return np.array(res)

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

    def update_LP(self):

        if (np.array([len(e) for e in self.successes_and_failures]) > 0.).all():
            # compute C, LP per module
            for i in range(self.n_relations):
                n_points = len(self.successes_and_failures[i])
                if n_points > 100:
                    sf = np.array(self.successes_and_failures[i])
                    self.C[i] = np.mean(sf[n_points // 2:])
                    self.LP[i] = np.abs(np.mean(sf[n_points // 2:]) - np.mean(sf[: n_points // 2]))

            # compute p
            if self.LP.sum() == 0:
                self.p = np.ones([self.n_relations]) / self.n_relations
            else:
                self.p = self.LP / self.LP.sum()

            if self.p.sum() > 1:
                self.p[np.argmax(self.p)] -= self.p.sum() - 1
            elif self.p.sum() < 1:
                self.p[-1] = 1 - self.p[:-1].sum()

    def sync(self):
        if self.curriculum_learning:
            self.p = MPI.COMM_WORLD.bcast(self.p, root=0)
            self.LP = MPI.COMM_WORLD.bcast(self.LP, root=0)
            self.C = MPI.COMM_WORLD.bcast(self.C, root=0)
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
        self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)

    def build_batch(self, batch_size):
        goal_ids = np.random.choice(np.arange(self.n_relations), p=self.p, size=batch_size)
        return goal_ids

    def init_stats(self):
        self.stats = dict()
        for i in np.arange(self.n_relations):
            self.stats['Eval_SR_{}'.format(i)] = []
            self.stats['Av_Rew_{}'.format(i)] = []
        for i in range(self.n_relations):
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
        for g_id in np.arange(self.n_relations):
            self.stats['Eval_SR_{}'.format(g_id)].append(av_res[g_id])
            self.stats['Av_Rew_{}'.format(g_id)].append(av_rew[g_id])

        for i in range(self.n_relations):
            self.stats['B_{}_LP'.format(i)].append(self.LP[i])
            self.stats['B_{}_C'.format(i)].append(self.C[i])
            self.stats['B_{}_p'.format(i)].append(self.p[i])
