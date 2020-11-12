import numpy as np
from scipy.linalg import block_diag


class her_sampler:
    def __init__(self, args, continuous=False, reward_func=None):
        self.replay_strategy = args.replay_strategy
        self.replay_k = args.replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + self.replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        self.continuous = continuous  # whether to use semantic configurations or continuous goals
        self.obj_inds = np.array([np.arange(i * 3, (i+1) * 3) for i in range(len(args.object_inds))])

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # replace goal with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        # transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward

        if self.continuous:
            # multi-criteria her
            n_replay = her_indexes[0].size
            number_of_blocks_to_replay = np.random.choice([1, 2, 3], size=n_replay)
            for i in range(n_replay):
                ids = np.random.choice([0, 1, 2], size=number_of_blocks_to_replay[i], replace=False)
                obs_inds = np.concatenate(self.obj_inds[ids])
                transitions['g'][her_indexes[0][i]][obs_inds] = future_ag[i][obs_inds]
            transitions['r'] = np.expand_dims(compute_reward(transitions['ag_next'], transitions['g'], None), 1)
            return transitions
        else:
            transitions['g'][her_indexes] = future_ag
            transitions['r'] = np.expand_dims(np.array([self.reward_func(ag_next, g, None) for ag_next, g in zip(transitions['ag_next'],
                                                                                                                 transitions['g'])]), 1)
            return transitions

def compute_reward(ag, g, info):
    dists = []
    for i in range(3):
        dists.append(np.linalg.norm(g[:, i * 3: (i+1) * 3] - ag[:, i * 3: (i+1) * 3], axis=1))
    return (np.array(dists) < 0.05).sum(axis=0).astype(np.float32)
