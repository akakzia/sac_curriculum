import torch
from rl_modules.sac_agent2 import SACAgent
from arguments import get_args
import env
import gym
import numpy as np
from utils import generate_goals, generate_all_goals_in_goal_space
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
import  random
from mpi4py import MPI
import torch
import pickle
from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# process the inputs
def normalize_goal(g, g_mean, g_std, args):
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    return g_norm


def normalize(o, o_mean, o_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    return o_norm


def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    return inputs

def get_env_params(env):
    obs = env.reset()

    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

def sample_vae(vae, inst_to_one_hot, config_init, sentence, n=1):

    one_hot = np.expand_dims(np.array(inst_to_one_hot[sentence.lower()]), 0)
    c_i = np.expand_dims(config_init, 0)
    one_hot = np.repeat(one_hot, n, axis=0)
    c_i = np.repeat(c_i, n, axis=0)
    c_i, s = torch.Tensor(c_i).to(device), torch.Tensor(one_hot).to(device)
    x = (vae.inference(c_i, s, n=n).detach().numpy() > 0.5).astype(np.int)

    return x

def sample_vae_logic(vae, inst_to_one_hot, config_init, expression, dict_goals, n=30):

    expression_type = expression[0]

    if isinstance(expression[1], str):
        x = sample_vae(vae, inst_to_one_hot, config_init, deepcopy(expression[1]), n=n)
        x_strs = [str(xi) for xi in x]
        set_1 = set(x_strs)
    elif isinstance(expression[1], list):
        set_1 = sample_vae_logic(vae, inst_to_one_hot, config_init, deepcopy(expression[1]), dict_goals=dict_goals)
    else:
        raise NotImplementedError

    if expression_type == 'not':
        return set(dict_goals.keys()).difference(set_1)
    elif expression_type in ['and', 'or']:
        if isinstance(expression[2], str):
            x = sample_vae(vae, inst_to_one_hot, config_init, deepcopy(expression[2]), n=n)
            x_strs = [str(xi) for xi in x]
            set_2 = set(x_strs)
        elif isinstance(expression[2], list):
            set_2 = sample_vae_logic(vae, inst_to_one_hot, config_init, deepcopy(expression[2]), dict_goals=dict_goals)
        else:
            raise NotImplementedError

        if expression_type == 'and':
            return set_1.intersection(set_2)
        elif expression_type == 'or':
            return set_1.union(set_2)
    else:
        raise NotImplementedError



def rollout(vae, sentences, inst_to_one_hot, dict_goals, env, policy, env_params, inits, goals, self_eval, true_eval, biased_init=False, animated=False):
    episodes = []
    observation = env.unwrapped.reset_goal(np.array(goals[i]), init=inits[i], biased_init=biased_init)
    env.render()


    for sentence in sentences:
        print(sentence)
        # sentence = input()
        expression = ['or', ['and', 'put red close_to green', 'put blue close_to green' ], ['and', 'put red above green', ['not', 'put blue close_to green']]]
        expression = ['and', 'put red above blue', 'put blue on_top_of green' ]
        expression = ['and', 'put red above green', 'put red above blue']
        if True:#sentence.lower() in inst_to_one_hot.keys():
            # goal = sample_vae(vae, inst_to_one_hot, observation['achieved_goal'], sentence, n=1).flatten().astype(np.float32)

            goals_str = sample_vae_logic(vae, inst_to_one_hot, observation['achieved_goal'], expression, dict_goals)

            goal = dict_goals[np.random.choice(list(goals_str))]
            env.unwrapped.target_goal = goal.copy()
            observation = env.unwrapped._get_obs()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            ep_obs, ep_ag, ep_g, ep_actions, ep_success = [], [], [], [], []

            # start to collect samples
            for t in range(env_params['max_timesteps']):
                # run policy
                no_noise = self_eval or true_eval
                action = policy.act(obs.copy(), ag.copy(), g.copy(), no_noise)
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())

                # feed the actions into the environment
                if animated:
                    env.render()
                observation_new, _, _, info = env.step(action)
                obs = observation_new['observation']
                ag = observation_new['achieved_goal']
            ep_ag.append(ag.copy())

            episode = dict(g=np.array(ep_g),
                           ag=np.array(ep_ag))
            episodes.append(episode)
        else:
            print('Wrong sentence.')

if __name__ == '__main__':
    num_eval = 10
    path = '/home/flowers/Downloads/test/'
    model_path = path + 'model_600.pt'

    with open(path + 'config.json', 'r') as f:
        params = json.load(f)
    args = SimpleNamespace(**params)

    # Make the environment
    env = gym.make(args.env_name)

    # set random seeds for reproduce
    args.seed = np.random.randint(1e6)
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    args.env_params = get_env_params(env)

    goal_sampler = GoalSampler(args)

    # create the sac agent to interact with the environment
    if args.agent == "SAC":
        policy = SACAgent(args, env.compute_reward, goal_sampler)
        policy.load(model_path, args)
    else:
        raise NotImplementedError

    # def rollout worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    eval_goals = goal_sampler.valid_goals
    inits = [None] * len(eval_goals)
    all_results = []

    with open(path + 'vae_model.pkl', 'rb') as f:
        vae = torch.load(f)


    with open(path + 'inst_to_one_hot.pkl', 'rb') as f:
        inst_to_one_hot = pickle.load(f)

    with open(path + 'sentences_list.pkl', 'rb') as f:
        sentences = pickle.load(f)

    all_goals = generate_all_goals_in_goal_space()
    dict_goals = dict(zip([str(g) for g in all_goals], all_goals))
    dataset = []
    ids_objs = [np.array([10, 11, 12]), np.array([25, 26, 27]), np.array([40, 41, 42])]
    for i in range(5000 // 35 + 1):
        print(len(dataset) / 5000)
        # uncomment here to run normal eval
        episodes = rollout_worker.generate_rollout(inits, eval_goals, self_eval=True, true_eval=True, animated=False)
        # episodes = rollout(vae, sentences, inst_to_one_hot, dict_goals, env, policy, args.env_params, inits, eval_goals, self_eval=True, true_eval=True, animated=True)

        results = np.array([str(e['g'][0]) == str(e['ag'][-1]) for e in episodes]).astype(np.int)
        for e in episodes:
            obj_pos_init = np.array([e['obs'][0][ido] for ido in ids_objs]).flatten()
            obj_pos_final = np.array([e['obs'][-1][ido] for ido in ids_objs]).flatten()

            dataset.append(np.array([e['ag'][0], e['ag'][-1], obj_pos_init, obj_pos_final]))

    with open('/home/flowers/Desktop/dataset_config.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    results = np.array(all_results)
    print('Av Success Rate: {}'.format(results.mean()))

