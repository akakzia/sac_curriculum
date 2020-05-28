import numpy as np
from datetime import datetime
import itertools
import os
import json


def above_to_close(vector):
    """
    Given a configuration of above objects, determines a configuration of close objects
    :param vector:
    :return:
    """
    size = len(vector)
    res = np.zeros(size//2)
    for i in range(size//2):
        if vector[2*i] == 1. or vector[2*i+1] == 1.:
            res[i] = 1.
    return tuple(res)


def valid(vector):
    """
    Determines whether an above configuration is valid or not
    :param vector:
    :return:
    """
    size = len(vector)
    if sum(vector) > 2:
        return False
    else:
        """can't have x on y and y on x"""
        for i in range(size//2):
            if vector[2*i] == 1. and vector[2*i] == vector[2*i+1]:
                return False
        """can't have two blocks on one blocks"""
        if (vector[0] == 1. and vector[0] == vector[-1]) or \
                (vector[1] == 1. and vector[1] == vector[3]) or (vector[2] == 1. and vector[2] == vector[4]):
            return False
    return True


def one_above_two(vector):
    """
    Determines whether one block is above two blocks
    """
    if (vector[0] == 1. and vector[0] == vector[2]) or \
            (vector[1] == 1. and vector[1] == vector[-2]) or (vector[3] == 1. and vector[3] == vector[-1]):
        return True
    return False


stack_three_list = [(1., 1., 0., 1., 0., 0., 1., 0., 0.), (1., 0., 1., 0., 1., 0., 0., 0., 1.),
                    (1., 1., 0., 0., 1., 1., 0., 0., 0.), (1., 0., 1., 1., 0., 0., 0., 1., 0.),
                    (0., 1., 1., 0., 0., 1., 0., 0., 1.), (0., 1., 1., 0., 0., 0., 1., 1., 0.)]


def generate_all_goals_in_goal_space():
    goals = []
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                for d in [0, 1]:
                    for e in [0, 1]:
                        for f in [0, 1]:
                            for g in [0, 1]:
                                for h in [0, 1]:
                                    for i in [0, 1]:
                                        goals.append([a, b, c, d, e, f, g, h, i])

    return np.array(goals)


def generate_goals(nb_objects=3, sym=1, asym=1):
    """
    generates all the possible goal configurations whether feasible or not, then regroup them into buckets
    :return:
    """
    buckets = {0: [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                   (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],

                1: [(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],

                2: [(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                    (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)],

                3: [(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                    (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0), (1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                     (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)],

                4:  [(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0), (0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
                     (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                     (1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                     ]}
    return buckets

def get_instruction():
    buckets = generate_goals(nb_objects=3, sym=1, asym=1)

    all_goals = generate_all_goals_in_goal_space().astype(np.float32)
    valid_goals = []
    for k in buckets.keys():
        # if k < 4:
        valid_goals += buckets[k]
    valid_goals = np.array(valid_goals)
    all_goals = np.array(all_goals)
    num_goals = all_goals.shape[0]
    all_goals_str = [str(g) for g in all_goals]
    valid_goals_str = [str(vg) for vg in valid_goals]

    # initialize dict to convert from the oracle id to goals and vice versa.
    # oracle id is position in the all_goal array
    g_str_to_oracle_id = dict(zip(all_goals_str, range(num_goals)))
    valid_goals_oracle_ids = np.array([g_str_to_oracle_id[str(vg)] for vg in valid_goals])


    instructions = ['Bring blocks away_from each_other',
                    'Bring blue close_to green and red far',
                    'Bring blue close_to red and green far',
                    'Bring green close_to red and blue far',
                    'Bring blue close_to red and green',
                    'Bring green close_to red and blue',
                    'Bring red close_to green and blue',
                    'Bring all blocks close',
                    'Stack blue on green and red far',
                    'Stack green on blue and red far',
                    'Stack blue on red and green far',
                    'Stack red on blue and green far',
                    'Stack green on red and blue far',
                    'Stack red on green and blue far',
                    'Stack blue on green and red close_from green',
                    'Stack green on blue and red close_from blue',
                    'Stack blue on red and green close_from red',
                    'Stack red on blue and green close_from blue',
                    'Stack green on red and blue close_from red',
                    'Stack red on green and blue close_from green',
                    'Stack blue on green and red close_from both',
                    'Stack green on blue and red close_from both',
                    'Stack blue on red and green close_from both',
                    'Stack red on blue and green close_from both',
                    'Stack green on red and blue close_from both',
                    'Stack red on green and blue close_from both',
                    'Stack green on red and blue',
                    'Stack red on green and blue',
                    'Stack blue on green and red',
                    'Stack green on blue and blue on red',
                    'Stack red on blue and blue on green',
                    'Stack blue on green and green on red',
                    'Stack red on green and green on blue',
                    'Stack green on red and red on blue',
                    'Stack blue on red and red on green',
                    ]
    words = ['stack', 'green', 'blue', 'on', 'red', 'and', 'close_from', 'both', 'far', 'close', 'all', 'bring', 'blocks', 'away_from', 'close_to']
    length = set()
    for s in instructions:
        if len(s) not in length:
            length.add(len(s.split(' ')))


    oracle_id_to_inst = dict()
    g_str_to_inst = dict()
    for g_str, oracle_id in g_str_to_oracle_id.items():
        if g_str in valid_goals_str:
            inst = instructions[valid_goals_str.index(g_str)]
        else:
            inst = ' '.join(np.random.choice(words, size=np.random.choice(list(length))))
        g_str_to_inst[g_str] = inst
        oracle_id_to_inst[g_str] = inst

    return oracle_id_to_inst, g_str_to_inst

def init_storage(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    # path to save the model
    logdir = os.path.join(args.save_dir, args.env_name + '_' + args.folder_prefix)
    if args.curriculum_learning:
        logdir = os.path.join(args.save_dir, '{}_curriculum_{}'.format(datetime.now(), args.architecture))
        if args.deepsets_attention:
            logdir += '_attention'
        if args.double_critic_attention:
            logdir += '_double'
    else:
        logdir = os.path.join(args.save_dir, '{}_no_curriculum_{}'.format(datetime.now(), args.architecture))
    # path to save evaluations
    model_path = os.path.join(logdir, 'models')
    bucket_path = os.path.join(logdir, 'buckets')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(bucket_path):
        os.mkdir(bucket_path)
    with open(os.path.join(logdir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    return logdir, model_path, bucket_path
