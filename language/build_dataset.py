from language.get_data import get_data
import numpy as np
import env
import gym
from utils import generate_goals, generate_all_goals_in_goal_space

def label_transitions(transitions, predicates, colors, n='all', add_abstract=True):
    data_configs, data_sentences = [], []
    # get all possible transitions between configs and corresponding sentence from dataset
    for transition in transitions:
        delta = transition[1] - transition[0]
        sentences = []
        if add_abstract:
            if transition[1][:3].sum() == 3:
                sentences.append('Put them all close')
                sentences.append('Get them all close')
            if transition[1][3:].sum() == 1:
                sentences.append('Make a tower')
                sentences.append('Build a tower')
                sentences.append('Stack some blocks')
                sentences.append('Make a stack of two')
                sentences.append('Make a tower of two')
                sentences.append('Build a tower of two')
                sentences.append('Build a stack of two')
                sentences.append('Stack two blocks')
                sentences.append('Make a construction')
                sentences.append('Build a construction')
                if int(predicates[int(np.argwhere(transition[1][3:]==1)) + 3][6]) == 0:
                    sentences.append('Put red on top')
                    sentences.append('Get red on top')
                if int(predicates[int(np.argwhere(transition[1][3:]==1)) + 3][6]) == 1:
                    sentences.append('Put green on top')
                    sentences.append('Get green on top')
                if int(predicates[int(np.argwhere(transition[1][3:]==1)) + 3][6]) == 2:
                    sentences.append('Put blue on top')
                    sentences.append('Get blue on top')
            if transition[1][3:].sum() == 2:
                if transition[1][np.array([3,5])].sum() == 2 or transition[1][np.array([4, 7])].sum() == 2 or transition[1][np.array([8, 6])].sum() == 2:
                    sentences.append('Make a pyramid')
                    sentences.append('Build a pyramid')
                    sentences.append('Make a construction')
                    sentences.append('Build a construction')
                else:
                    sentences.append('Make a tower')
                    sentences.append('Build a tower')
                    sentences.append('Stack some blocks')
                    sentences.append('Make a stack of three')
                    sentences.append('Make a tower of three')
                    sentences.append('Build a stack of three')
                    sentences.append('Build a tower of three')
                    sentences.append('Stack two blocks')
                    sentences.append('Make a construction')
                    sentences.append('Build a construction')
                    if (transition[1][3] == 1 and transition[1][7] == 1) or (transition[1][5] == 1 and transition[1][8] == 1):
                        sentences.append('Put red on top')
                        sentences.append('Get red on top')
                    if (transition[1][4] == 1 and transition[1][5] == 1) or (transition[1][7] == 1 and transition[1][6] == 1):
                        sentences.append('Put green on top')
                        sentences.append('Get green on top')
                    if (transition[1][6] == 1 and transition[1][3] == 1) or (transition[1][8] == 1 and transition[1][4] == 1):
                        sentences.append('Put blue on top')
                        sentences.append('Get blue on top')
        abstract_sentences = sentences.copy()
        for i in range(len(predicates)):
            if delta[i] != 0:
                p = predicates[i]
                words = p.split('_')
                for j in range(len(words)):
                    try:
                        words[j] = colors[words[j]]
                    except:
                        pass
                positive = delta[i] == 1
                if words[0] == 'close':
                    if positive:
                        sentences.append('Put {} close_to {}'.format(words[1], words[2]))
                        sentences.append('Get {} close_to {}'.format(words[1], words[2]))
                        sentences.append('Put {} close_to {}'.format(words[2], words[1]))
                        sentences.append('Get {} close_to {}'.format(words[2], words[1]))
                        sentences.append('Get {} and {} close_from each_other'.format(words[1], words[2]))
                        sentences.append('Get {} and {} close_from each_other'.format(words[2], words[1]))
                        sentences.append('Bring {} and {} together'.format(words[1], words[2]))
                        sentences.append('Bring {} and {} together'.format(words[2], words[1]))
                    else:
                        sentences.append('Put {} far_from {}'.format(words[1], words[2]))
                        sentences.append('Get {} far_from {}'.format(words[1], words[2]))
                        sentences.append('Put {} far_from {}'.format(words[2], words[1]))
                        sentences.append('Get {} far_from {}'.format(words[2], words[1]))
                        sentences.append('Get {} and {} far_from each_other'.format(words[1], words[2]))
                        sentences.append('Get {} and {} far_from each_other'.format(words[2], words[1]))
                        sentences.append('Bring {} and {} apart'.format(words[1], words[2]))
                        sentences.append('Bring {} and {} apart'.format(words[2], words[1]))
                elif words[0] == 'above':
                    if positive:
                        sentences.append('Put {} above {}'.format(words[1], words[2]))
                        sentences.append('Put {} on_top_of {}'.format(words[1], words[2]))
                        sentences.append('Put {} under {}'.format(words[2], words[1]))
                        sentences.append('Put {} below {}'.format(words[2], words[1]))
                    else:
                        sentences.append('Remove {} from {}'.format(words[1], words[2]))
                        sentences.append('Remove {} from_above {}'.format(words[1], words[2]))
                        sentences.append('Remove {} from_under {}'.format(words[2], words[1]))
                        sentences.append('Remove {} from_below {}'.format(words[2], words[1]))
                        sentences.append('Put {} and {} on_the_same_plane'.format(words[1], words[2]))
                        sentences.append('Put {} and {} on_the_same_plane'.format(words[2], words[1]))
                else:
                    raise NotImplementedError
        if len(sentences) != 0:
            if n == 'all':
                data_configs += [transition.copy()] * len(sentences)
                data_sentences += sentences
            else:
                if n > len(sentences):
                    data_sentences += sentences
                    data_configs += [transition.copy()] * len(sentences)
                else:
                    if len(abstract_sentences) > 0:
                        if np.random.rand() < 0.5:
                            data_sentences += np.array(np.random.choice(abstract_sentences, size=n, replace=False)).flatten().tolist()
                        else:
                            data_sentences += np.array(np.random.choice(sentences, size=n, replace=False)).flatten().tolist()
                    else:
                        data_sentences += np.array(np.random.choice(sentences, size=n, replace=False)).flatten().tolist()
                    data_configs += [transition.copy()] * n
    return data_configs.copy(), data_sentences.copy()


def get_dataset(binary=True):
    unique_reached_config_transitions, predicates, \
    predicate_to_id, id_to_predicate, colors = get_data(binary)

    # get synthetic valid goals
    all_valid_goals = []
    buckets = generate_goals()
    for b in buckets.values():
        for g in b:
            all_valid_goals.append(np.array(g))
    all_valid_str = [str(vg) for vg in all_valid_goals]

    # add to valid goals all goals that are reached in the dataset
    all_configs_dataset = np.concatenate([unique_reached_config_transitions[:, 0, :], unique_reached_config_transitions[:, 1, :]], axis=0)
    all_configs_dataset_str = [str(ac) for ac in all_configs_dataset.astype(np.int)]
    unique_str, idx = np.unique(all_configs_dataset_str, return_index=True)

    for i_s, s in enumerate(unique_str):
        if s not in all_valid_str:
            all_valid_goals.append(all_configs_dataset[idx[i_s]].astype(np.int))

    # compute all possible transitions between valid goals
    init_finals = []
    for g in all_valid_goals:
        for g2 in all_valid_goals:
            if not (g==g2).all():
                init_finals.append([g, g2])
    init_finals = np.array(init_finals)

    # construct dataset language
    data_configs, data_sentences = label_transitions(unique_reached_config_transitions, predicates, colors, n=1)
    all_possible_configs, all_possible_sentences = label_transitions(init_finals, predicates, colors, n='all')


    data_configs = np.array(data_configs[:5000])
    data_sentences = data_sentences[:5000]
    all_possible_configs = np.array(all_possible_configs)

    if binary:
        data_configs = data_configs[:, :2, :]
        data_continuous = None
    else:
        data_continuous = data_configs[:, 2:, :]
        data_configs = data_configs[:, :2, :]
    return data_configs.astype(np.int), data_sentences, data_continuous, all_possible_configs.astype(np.int), all_possible_sentences