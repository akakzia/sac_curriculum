import pickle
from torch.utils.data import Dataset
import numpy as np
import os
from itertools import combinations, permutations

o_ids = [0, 1, 2, 3, 10, 11, 12, 13, 14, 18, 22, 26]

class ConfigDataset(Dataset):
    def __init__(self, initial_configs, initial_states, final_configs):

        assert initial_configs.shape[0] == initial_states.shape[0] == final_configs.shape[0]

        inds = np.arange(initial_configs.shape[0])
        np.random.shuffle(inds)

        self.init_configs = initial_configs[inds].astype(np.float32)
        self.init_states = initial_states[inds].astype(np.float32)
        self.fin_configs = final_configs[inds].astype(np.float32)

        self.idx = inds

    def __getitem__(self, index):
        return self.init_configs[index], self.init_states[index], self.fin_configs[index]

    def __len__(self):
        return self.idx.shape[0]

def load_dataset():
    """
    Load the dataset of configurations and geometric states
    """
    path_states_configs = os.path.join(os.getcwd(), 'proximal_generator', 'data', 'dataset_one_block.pkl')
    with open(path_states_configs, 'rb') as f:
        init_configs, init_states, final_configs, final_states, init_to_finals = pickle.load(f)

    return init_configs, init_states, final_configs, final_states, init_to_finals

def split_data(init_configs, init_states, final_configs, init_to_finals):
    nb_test_goals = 100

    # Test set
    remove_str = np.random.choice([str(e) for e in np.unique(init_configs, axis=0)], size=nb_test_goals)
    # remove_str = np.random.choice([str(e) for e in init_to_finals.keys()], size=nb_test_goals)

    # Find indices of the different sets in the total dataset.
    set_ids = [[] for _ in range(2)]
    for i, c_i in enumerate(init_configs):
        used = False

        if not used:
            if str(c_i) in remove_str:
                set_ids[1].append(i)
            else:
                set_ids[0].append(i)


    valid_ids = np.array(set_ids[0])
    dataset = ConfigDataset(init_configs[valid_ids], init_states[valid_ids], final_configs[valid_ids])

    return set_ids, dataset

def get_per_object_predicate_indexes(nb_objects):
    map_pairs = list(combinations(np.arange(nb_objects), 2)) + list(permutations(np.arange(nb_objects), 2))
    res = []
    for i in range(nb_objects):
        temp = []
        for k, p in enumerate(map_pairs):
            if i in set(p):
                temp.append(k)
        res.append(temp)
    return res

def get_graph_structure(n):
    """ Given the number of blocks (nodes), returns :
    edges: in the form [to, from]
    incoming_edges: for each node, the indexes of the incoming edges
    predicate_ids: the ids of the predicates takes for each edge """
    map_list = list(combinations(np.arange(n), 2)) + list(permutations(np.arange(n), 2))
    edges = list(permutations(np.arange(n), 2))
    obj_ids = np.arange(n)
    n_comb = n * (n-1) // 2

    incoming_edges = []
    for obj_id in obj_ids:
        temp = []
        for i, pair in enumerate(permutations(np.arange(n), 2)):
            if obj_id == pair[0]:
                temp.append(i)
        incoming_edges.append(temp)

    predicate_ids = []
    for pair in permutations(np.arange(n), 2):
        ids_g = [i for i in range(len(map_list))
                 if (set(map_list[i]) == set(pair) and i < n_comb)
                 or (map_list[i] == pair and i >= n_comb)]
        predicate_ids.append(ids_g)

    return edges, incoming_edges, predicate_ids