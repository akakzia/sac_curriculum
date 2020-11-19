import numpy as np
from language.build_dataset import sentence_from_configuration
from utils import id_to_language, language_to_id


class her_sampler:
    def __init__(self, args, reward_func=None):
        self.replay_strategy = args.replay_strategy
        self.replay_k = args.replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + args.replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

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
        n_replay = her_indexes[0].size
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # replace goal with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        future_lg = np.array([sentence_from_configuration(future_ag[i]) for i in range(n_replay)])
        transitions['lg_ids'][her_indexes[0]] = np.array([language_to_id[f_lg] for f_lg in future_lg])
        transitions['r'] = np.expand_dims(compute_reward_language(transitions['ag_next'], transitions['lg_ids']), 1)

        return transitions


def compute_reward_language(ags, lg_ids):
    lgs = [id_to_language[lg_id] for lg_id in lg_ids]
    r = np.array([lg in sentence_from_configuration(ag, all=True) for ag, lg in zip(ags, lgs)]).astype(np.float32)
    return r
