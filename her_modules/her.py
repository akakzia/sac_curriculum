import numpy as np
from scipy.linalg import block_diag
from language.build_dataset import sentence_from_configuration
from utils import id_to_language, language_to_id


MC_MASK = True

class her_sampler:
    def __init__(self, args, reward_func=None):
        self.replay_strategy = args.replay_strategy
        self.replay_k = args.replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + args.replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        self.continuous = args.algo == 'continuous'  # whether to use semantic configurations or continuous goals
        self.language = args.algo == 'language'
        self.multi_criteria_her = args.multi_criteria_her
        self.obj_ind = np.array([np.arange(i * 3, (i + 1) * 3) for i in range(3)])
        # self.semantic_ids = np.array([np.array([0, 1, 3, 4, 5, 6]), np.array([0, 2, 3, 4, 7, 8]), np.array([1, 2, 5, 6, 7, 8])])
        self.semantic_ids = np.array([np.array([0, 3, 4]), np.array([1, 5, 6]), np.array([2, 7, 8])])
        self.mask_ids = np.array([np.array([0, 3, 4]), np.array([1, 5, 6]), np.array([2, 7, 8])])
        self.mask_p = args.mask_p

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        if not self.continuous:
            # her idx
            if self.multi_criteria_her:
                if MC_MASK:
                    count = 0
                    np.random.shuffle(self.mask_ids)
                    for sub_goal, sub_mask in zip(self.semantic_ids, self.mask_ids):
                        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
                        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
                        future_offset = future_offset.astype(int)
                        future_t = (t_samples + 1 + future_offset)[her_indexes]
                        # Replace
                        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
                        transition_goals = transitions['g'][her_indexes]
                        transition_goals[:, sub_goal] = future_ag[:, sub_goal]
                        transitions['g'][her_indexes] = transition_goals

                        # Mask sub goals
                        if count < 1:
                            count += 1
                            mask_indexes = np.where(np.random.uniform(size=batch_size) < self.mask_p)
                            transition_masks = transitions['masks'][mask_indexes]
                            transition_masks[:, sub_mask] = 1.
                            transitions['masks'][mask_indexes] = transition_masks
                else:
                    for sub_goal in self.semantic_ids:
                        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
                        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
                        future_offset = future_offset.astype(int)
                        future_t = (t_samples + 1 + future_offset)[her_indexes]
                        # Replace
                        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
                        transition_goals = transitions['g'][her_indexes]
                        transition_goals[:, sub_goal] = future_ag[:, sub_goal]
                        transitions['g'][her_indexes] = transition_goals
            else:
                her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
                n_replay = her_indexes[0].size
                future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
                future_offset = future_offset.astype(int)
                future_t = (t_samples + 1 + future_offset)[her_indexes]

                # replace goal with achieved goal
                future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
                transitions['g'][her_indexes] = future_ag
                # to get the params to re-compute reward
            transitions['r'] = np.expand_dims(np.array([compute_reward_masks(ag_next, g, mask) for ag_next, g, mask in zip(transitions['ag_next'],
                                                                                            transitions['g'], transitions['masks'])]), 1)
        else:
            if self.multi_criteria_her:
                for sub_goal in self.obj_ind:
                    her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
                    future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
                    future_offset = future_offset.astype(int)
                    future_t = (t_samples + 1 + future_offset)[her_indexes]
                    # Replace
                    future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
                    transition_goals = transitions['g'][her_indexes]
                    transition_goals[:, sub_goal] = future_ag[:, sub_goal]
                    transitions['g'][her_indexes] = transition_goals
            else:
                # her idx
                her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
                future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
                future_offset = future_offset.astype(int)
                future_t = (t_samples + 1 + future_offset)[her_indexes]

                # replace goal with achieved goal
                future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
                transitions['g'][her_indexes] = future_ag
            transitions['r'] = np.expand_dims(np.array([self.reward_func(ag_next, g, None) for ag_next, g in zip(transitions['ag_next'],
                                                                                                transitions['g'])]), 1)

        return transitions


def compute_reward_language(ags, lg_ids):
    lgs = [id_to_language[lg_id] for lg_id in lg_ids]
    r = np.array([lg in sentence_from_configuration(ag, all=True) for ag, lg in zip(ags, lgs)]).astype(np.float32)
    return r


def compute_reward_masks(ag, g, mask):
    reward = 0.
    # semantic_ids = np.array([np.array([0, 1, 3, 4, 5, 6]), np.array([0, 2, 3, 4, 7, 8]), np.array([1, 2, 5, 6, 7, 8])])
    semantic_ids = np.array([np.array([0, 3, 4]), np.array([1, 5, 6]), np.array([2, 7, 8])])
    ids = np.where(mask != 1.)[0]
    semantic_ids = [np.intersect1d(semantic_id, ids) for semantic_id in semantic_ids]
    for subgoal in semantic_ids:
        if (len(subgoal) > 0) and (ag[subgoal] == g[subgoal]).all():
            reward = reward + 1.
    return reward
