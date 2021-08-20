import numpy as np


class her_sampler:
    def __init__(self, args, reward_func=None):
        self.reward_type = args.reward_type
        self.replay_k = args.replay_k
        self.reward_func = reward_func
        self.her_p = 1 - (1. / (1 + args.replay_k))
        if args.replay_strategy not in ['final', 'future']:
            raise Exception("Unimplemented HER strategy : ", {args.replay_strategy})

        self.replay_strategy = args.replay_strategy

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        batch_size = batch_size_in_transitions
        rollout_batch_size = episode_batch['actions'].shape[0]

        if len(episode_batch) == batch_size_in_transitions:
            episode_ids = np.arange(0,len(episode_batch))
        else : 
            episode_ids = np.random.randint(0, rollout_batch_size, batch_size)

        # select which timesteps to be used
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_ids,t_samples].copy() for key in episode_batch.keys()}

        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.her_p)

        # future goal selection
        if self.replay_strategy == 'final':
            # fictive goal is the final achieved goal of the selected HER episodes
            future_ag = episode_batch['ag'][episode_ids[her_indexes],-1]
        else:
            # sample future achieved goals
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]

            # fictive goals are the selected future achieved goals
            future_ag = episode_batch['ag'][episode_ids[her_indexes], future_t]
        # replace goal with fictive goal
        transitions['g'][her_indexes] = future_ag
        transitions['r'] = np.expand_dims(np.array([self.reward_func(ag_next, g, None) for ag_next, g in zip(transitions['ag_next'],
                                                                                                transitions['g'])]), 1)
        return transitions
