import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""

ENERGY_BIAS = False


class MultiBuffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        self.energy_bias = ENERGY_BIAS

        # memory management
        self.sample_func = sample_func

        self.current_size = 0

        # create the buffer to store info
        self.buffer = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                       'ag': np.empty([self.size, self.T + 1, 9]),
                       'actions': np.empty([self.size, self.T, self.env_params['action']]),
                       'lg_ids': np.empty([self.size, self.T]).astype(np.int),
                       # 'language_goal': [None for _ in range(self.size)],
                       }
        if self.energy_bias:
            self.buffer['energy'] = np.empty([self.size])

        self.goal_ids = np.zeros([self.size])  # contains id of achieved goal (discovery rank)
        self.goal_ids.fill(np.nan)

        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        batch_size = len(episode_batch)
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)

            for i, e in enumerate(episode_batch):
                # store the informations
                self.buffer['obs'][idxs[i]] = e['obs']
                self.buffer['ag'][idxs[i]] = e['ag']
                self.buffer['actions'][idxs[i]] = e['act']
                self.buffer['lg_ids'][idxs[i]] = e['lg_ids']
                if self.energy_bias:
                    if len(set([str(ag) for ag in e['ag']])) > 1:
                        self.buffer['energy'][idxs[i]] = 1
                    else:
                        self.buffer['energy'][idxs[i]] = 0

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            if self.energy_bias:
                energy = self.buffer['energy'][:self.current_size].astype(np.bool)
                ind_energy = np.argwhere(energy).flatten()
                print(ind_energy.size)
                if ind_energy.size * self.T < 10 * batch_size:
                    for key in self.buffer.keys():
                        if key == 'language_goal':
                            temp_buffers[key] = np.array([np.array(self.buffer[key][:self.current_size]) for _ in range(self.T)]).T
                            temp_buffers[key] = temp_buffers[key].astype('object')
                        elif key != 'energy':
                            temp_buffers[key] = self.buffer[key][:self.current_size]
                else:
                    ind_not_energy = np.argwhere(~energy).flatten()
                    buffer_ids = np.concatenate([ind_energy, np.random.choice(ind_not_energy, size=ind_energy.size, replace=False)])
                    for key in self.buffer.keys():
                        temp_buffers[key] = self.buffer[key][:self.current_size]
            else:
                for key in self.buffer.keys():
                    temp_buffers[key] = self.buffer[key][:self.current_size]

        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]


        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = [idx[0]]
        return idx
