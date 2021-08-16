import pickle
import numpy as np
import os
from arguments import get_args

o_ids = [[0, 1, 2, 3, 10, 11, 12, 13, 14, 18, 22, 26], [0, 4, 5, 6, 10, 14, 15, 16, 17, 19, 23, 27],
         [1, 4, 7, 8, 11, 15, 18, 19, 20, 21, 24, 28], [2, 5, 7, 9, 12, 16, 20, 22, 23, 24, 25, 29],
         [3, 6, 8, 9, 13, 17, 21, 25, 26, 27, 28, 29]]

def build_dataset(path, size):
    """
    Build a dataset of pairs of configurations from trajectories
    """
    path_states_configs = path
    with open(path_states_configs, 'rb') as f:
        trajectories = pickle.load(f)

    ids = np.random.choice(np.arange(len(trajectories)), size=size, replace=False)
    trajectories = [trajectories[i] for i in ids]

    init_configs = []
    init_states = []
    final_configs = []
    final_states = []
    init_to_finals = {}

    for trajectory in trajectories:
        configs, idxs = np.unique(trajectory['ag'], return_index=True, axis=0)
        for i in idxs:
            if i > 0:
                i_config = np.clip(trajectory['ag'][i-1] + 1, 0, 1)
                f_config = np.clip(trajectory['ag'][i] + 1, 0, 1)
                i_state = np.concatenate([trajectory['obs'][i-1][10 + 15*j: 13 + 15*j] for j in range(5)])
                f_state = np.concatenate([trajectory['obs'][i][10 + 15 * j: 13 + 15 * j] for j in range(5)])
                if str((i_config[o_ids[0]])) != str(f_config[o_ids[0]]):
                    init_configs.append(i_config)
                    final_configs.append(f_config)
                    init_states.append(i_state)
                    final_states.append(f_state)
                    try:
                        init_to_finals[str(i_config[o_ids[0]])].add(str(f_config[o_ids[0]]))
                    except KeyError:
                        init_to_finals[str(i_config[o_ids[0]])] = set([str(f_config[o_ids[0]])])
                    try:
                        init_to_finals[str(f_config[o_ids[0]])].add(str(i_config[o_ids[0]]))
                    except KeyError:
                        init_to_finals[str(f_config[o_ids[0]])] = set([str(i_config[o_ids[0]])])

    init_configs = np.array(init_configs)
    final_configs = np.array(final_configs)
    init_states = np.array(init_states)
    final_states = np.array(final_states)

    path_save = os.path.join(os.getcwd(), 'data', 'dataset_one_block.pkl')
    with open(path_save, 'wb') as f:
        pickle.dump((init_configs, init_states, final_configs,final_states, init_to_finals), f)

    return init_configs, init_states, final_configs, final_states, init_to_finals

if __name__ == '__main__':
    args = get_args()
    args.data_path = os.path.join(os.getcwd(), 'data', args.data_name + '.pkl')
    _ = build_dataset(path = args.data_path, size=10000)
