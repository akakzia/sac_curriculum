import torch
from rl_modules.sac_agent import SACAgent
import env
import gym
import numpy as np
from utils import generate_all_goals_in_goal_space
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
import  random
from mpi4py import MPI
import torch
import pickle
from copy import deepcopy
from language.utils import get_corresponding_sentences
from scipy.stats import entropy
from utils import generate_goals

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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



def rollout(sentence_generator, sentences, inst_to_one_hot, dict_goals, env, policy, env_params, inits, goals, self_eval, true_eval, biased_init=False, animated=False):

    score = []
    ags = []
    for sentence in sentences:
        # sentence = sentence.lower()
        # print('\nNew instruction: ', sentence)
        reached = False
        observation = env.reset() #.reset_goal(np.array(goals[i]), biased_init=biased_init)

        config_initial = observation['achieved_goal'].copy()
        if sentence.lower() in inst_to_one_hot.keys():
            counter = 0
            # print(sentence)
            while counter < 1:
                # goal = sample_vae(vae, inst_to_one_hot, observation['achieved_goal'], sentence).flatten()

                # goal = dict_goals[np.random.choice(list(goals_str))]
                # env.unwrapped.target_goal = goal.copy()
                observation = env.unwrapped._get_obs()
                obs = observation['observation']
                ag = observation['achieved_goal']
                g = observation['desired_goal']

                # start to collect samples
                for t in range(env_params['max_timesteps']):
                    # run policy
                    no_noise = self_eval or true_eval
                    action = policy.act(obs.copy(), ag.copy(), g.copy(), no_noise, language_goal=sentence)
                    # feed the actions into the environment
                    if animated:
                        env.render()
                    observation_new, _, _, info = env.step(action)
                    obs = observation_new['observation']
                    ag = observation_new['achieved_goal']
                counter += 1
                config_final = ag.copy()
                true_sentences = sentence_generator(config_initial, config_final)
                if sentence.lower() in true_sentences:
                    score.append(counter)
                    reached = True
                    # print('\tSuccess!')
                    break
                # else:
                    # print(sentence.lower())
                    # print('\tFailed. Trying again.')
        ags.append(ag)

        # else:
            # print('Wrong sentence.')

        if not reached:
            score.append(0)
            # print('\tFailed 5 times, Moving On.')

    return np.array(score), np.array(ags)

if __name__ == '__main__':
    num_eval = 50
    path = './results/baselines/eval_LB/'

    with open(path + 'config.json', 'r') as f:
        params = json.load(f)
    params['algo'] = 'language'
    params['self_eval_prob'] = 0.1
    params['object_inds'] = [None] * 3
    params['combinations_trick'] = False
    params['freq_target_update'] = 2
    params['multi_criteria_her'] = False
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

    eval_goals = goal_sampler.valid_goals
    inits = [None] * len(eval_goals)
    all_results = []

    with open(path + 'inst_to_one_hot.pkl', 'rb') as f:
        inst_to_one_hot = pickle.load(f)

    with open(path + 'sentences_list.pkl', 'rb') as f:
        sentences = pickle.load(f)

    sentence_generator = get_corresponding_sentences
    all_goals = generate_all_goals_in_goal_space()
    dict_goals = dict(zip([str(g) for g in all_goals], all_goals))

    all_entropies = []
    all_counts = []

    for seed in [2, 4]:
        print('SEED', seed)
        # Load policy
        model_path = path + 'policy_models/model{}.pt'.format(seed + 1)
        # create the sac agent to interact with the environment
        if args.agent == "SAC":
            policy = SACAgent(args, env.compute_reward, goal_sampler)
            policy.load(model_path, args)
        else:
            raise NotImplementedError

        # Initialize Rollout Worker
        rollout_worker = RolloutWorker(env, policy, goal_sampler, args)

        scores = []
        all_ags = []
        for i in range(num_eval):
            if (i + 1) % 10 == 0:
                print(i+1)
            score, ags = rollout(sentence_generator,
                                 sentences,
                                 inst_to_one_hot,
                                 dict_goals,
                                 env,
                                 policy,
                                 args.env_params,
                                 inits,
                                 eval_goals,
                                 self_eval=True,
                                 true_eval=True,
                                 animated=False)
            scores.append(score)
            all_ags.append(ags)

        # compute entropy of ags
        # all_ags = np.array(all_ags)
        # set_ags = set()
        # for i in range(all_ags.shape[0]):
        #     for j in range(all_ags.shape[1]):
        #         set_ags.add(str(all_ags[i, j]))
        all_ags = np.array(all_ags)
        buckets = generate_goals()
        all_ags_for_set = []
        for i in buckets.keys():
            all_ags_for_set += buckets[i]
        all_ags_for_set = np.array(all_ags_for_set)
        ags = sorted(list(set([str(ag) for ag in all_ags_for_set])))
        # ags = sorted(list(set_ags))
        entropies = []
        diversity_counts = []
        for s in range(len(sentences)):
            counts = dict(zip(ags, [0] * len(ags)))
            total = 0
            for i in range(num_eval):
                if str(all_ags[i, s]) in counts.keys():
                    counts[str(all_ags[i, s])] += 1
                    total += 1
            probs = []
            for k in ags:
                probs.append(counts[k] / total)
            probs = np.array(probs)
            if probs.sum() != 1:
                probs[np.argmax(probs)] += (1 - probs.sum())
                if probs.sum() != 1:
                    print(probs)
            # assert sum(probs) == 1
            entropies.append(entropy(probs))
            diversity_counts.append(np.argwhere(probs > 0).size)

        all_entropies.append(entropies)
        all_counts.append(diversity_counts)

        np.savetxt('/home/flowers/Desktop/entropies24.txt', np.array(all_entropies))
        np.savetxt('/home/flowers/Desktop/counts24.txt', np.array(all_counts))
        # ratio_success = []
        # av_not_0 = []
        # ratio_first_shot = []
        # for r in np.array(scores):
        #     inds_not_0 = np.argwhere(r > 0).flatten()
        #     ratio_success.append(inds_not_0.size / r.size)
        #     ratio_first_shot.append(np.argwhere(r == 1).flatten().size / r.size)
        #     av_not_0.append(r[inds_not_0].mean())
        # print('Success rate (5 attempts): ', np.mean(ratio_success))
        # print('Success rate (first_shot): ', np.mean(ratio_first_shot))
        # print('When success, average nb of attempts: ', np.mean(av_not_0))




