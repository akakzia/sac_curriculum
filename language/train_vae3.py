import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from collections import defaultdict
from language.utils import analyze_inst, Vocab, OneHotEncoder, ConfigLanguageDataset
from language.vae import ContextVAE
from language.build_dataset import get_dataset
import numpy as np
import pickle
import env
import gym

def get_test_sets(configs, sentences, set_inds, states, all_possible_configs, str_to_index):

    configs = configs[set_inds]
    states = states[set_inds]
    sentences = np.array(sentences)[set_inds].tolist()

    config_init_and_sentence = []
    for i in range(configs.shape[0]):
        config_init_and_sentence.append(str(configs[i, 0]) + sentences[i])
    unique, idx, idx_in_array = np.unique(np.array(config_init_and_sentence), return_inverse=True, return_index=True)

    train_inits = []
    train_sents = []
    train_finals_dataset = []
    train_finals_possible = []
    train_cont_inits = []
    for i, i_array in enumerate(np.arange(len(sentences))):
        train_inits.append(configs[i_array, 0])
        train_sents.append(sentences[i_array])
        train_cont_inits.append(states[i_array, 0])
        # find all final configs compatible with init and sentence (in dataset and in possible configs)
        init_sent_str = config_init_and_sentence[i_array]
        # find all possible final configs (from dataset + synthetic)
        final_confs = all_possible_configs[str_to_index[init_sent_str], 1]
        final_str = [str(c) for c in final_confs]
        # find all possible final configs (from dataset)
        id_in_unique = unique.tolist().index(init_sent_str)
        idx_finals = np.argwhere(idx_in_array == id_in_unique).flatten()
        unique_final = np.unique(configs[idx_finals, 1], axis=0)
        # check that the one found in dataset are indeed in all the final configs possible
        # for c in unique_final:
        #     if str(c) not in final_str:
        #         print(str(c))
        #         stop = 1
        train_finals_possible.append(final_str)
        c_f_dataset = [str(c) for c in unique_final]
        train_finals_dataset.append(c_f_dataset)

    return train_inits, train_sents, train_finals_dataset, train_finals_possible, train_cont_inits

def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    configs, sentences, states, all_possible_configs, all_possible_sentences = get_dataset(binary=False)

    fake_states = np.concatenate([states[:, 0, :], states[:, 1, :]], axis=0)
    s_max = fake_states.max(axis=0)
    s_min = fake_states.min(axis=0)

    states = (states - s_min) / (s_max - s_min)

    set_sentences = set(sentences)
    split_instructions, max_seq_length, word_set = analyze_inst(set_sentences)
    vocab = Vocab(word_set)
    one_hot_encoder = OneHotEncoder(vocab, max_seq_length)
    inst_to_one_hot = dict()
    for s_instr in split_instructions:
        inst_to_one_hot[' '.join(s_instr)] = one_hot_encoder.encode(s_instr)


    all_str = ['start' + str(c[0]) + s + str(c[1]) + 'end' for c, s in zip(configs, sentences)]
    all_possible_configs_str = [str(c[0]) + s for c, s in zip(all_possible_configs, all_possible_sentences)]

    # test particular combinations of init, sentence, final
    # this tests the extrapolation to different final states than the ones in train set
    remove1 = [[[0, 0, 1, 0, 0, 0, 0, 0, 0], 'Get blue and red far_from each_other', [0, 1, 0, 0, 0, 0, 0, 0, 0]],
               [[0, 0, 1, 0, 0, 0, 0, 0, 0], 'Put blue above green', [1, 0, 0, 0, 1, 0, 0, 0, 0]],
               [[0, 0, 0, 0, 0, 0, 0, 0, 0], 'Get blue close_to red', [0, 0, 1, 0, 0, 0, 0, 0, 0]],
               [[0, 0, 0, 0, 0, 0, 0, 0, 0], 'Bring red and green together', [1, 1, 0, 0, 0, 0, 0, 0, 0]],
               [[0, 0, 1, 0, 0, 0, 0, 0, 0], 'Put green on_top_of blue', [1, 0, 0, 1, 0, 0, 0, 0, 0]]]
    remove1_str = ['start' + str(np.array(r[0])) + r[1] + str(np.array(r[2])) for r in remove1]

    remove2 = [[[0, 1, 0, 0, 0, 0, 0, 0, 0], 'Get blue close_to red'],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0], 'Put green above red']]
    remove2_str = ['start' +  str(np.array(r[0])) + r[1] for r in remove2]

    remove3 = [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0]]
    remove3_str = ['start' + str(np.array(r)) for r in remove3]

    remove4 = ['Put green on_top_of red', 'Put blue under green', 'Bring red and blue apart']
    remove4_str = remove4.copy()

    # what about removing all of one final state, or combinations of sentence and final state, or init and final ?

    set_inds = [[] for _ in range(6)]
    for i, s in enumerate(all_str):

        to_remove = False

        used = False
        for s1 in remove1_str:
            if s1 in s:
                set_inds[1].append(i)
                used = True
                break

        if not used:
            for s2 in remove2_str:
                if s2 in s:
                    set_inds[2].append(i)
                    used = True
                    break

        if not used:
            for s3 in remove3_str:
                if s3 in s:
                    does_s_also_contains_s_from_r4 = False
                    for s4 in remove4_str:
                        if s3 + s4 in s:
                            does_s_also_contains_s_from_r4 = True
                    used = True
                    if not does_s_also_contains_s_from_r4:
                        set_inds[3].append(i)
                    else:
                        set_inds[5].append(i)
                    break

        if not used:
            for s4 in remove4_str:
                if s4 in s:
                    does_s_also_contains_s_from_r3 = False
                    for s3 in remove3_str:
                        if s3 + s4 in s:
                            does_s_also_contains_s_from_r3 = True
                    used = True
                    if not does_s_also_contains_s_from_r3:
                        set_inds[4].append(i)
                    else:
                        set_inds[5].append(i)
                    break

        if not used and not to_remove:
            set_inds[0].append(i)

    assert np.sum([len(ind) for ind in set_inds]) == len(all_str)

    # dictionary translating string of init config and sentence to all possible final config (id in all_possible_configs)
    # including the ones in the dataset, but also other synthetic ones. This is used for evaluation
    str_to_index = dict()
    for i_s, s in enumerate(all_possible_configs_str):
        if s in str_to_index.keys():
            str_to_index[s].append(i_s)
        else:
            str_to_index[s] = [i_s]
    for k in str_to_index.keys():
        str_to_index[k] = np.array(str_to_index[k])

    train_test_data = get_test_sets(configs, sentences, set_inds[0], states, all_possible_configs, str_to_index)
    test_data = [get_test_sets(configs, sentences, set_ids, states, all_possible_configs, str_to_index) for set_ids in set_inds[1:]]
    valid_inds = np.array(set_inds[0])
    dataset = ConfigLanguageDataset(configs[valid_inds],
                                    np.array(sentences)[valid_inds].tolist(),
                                    states[valid_inds],
                                    inst_to_one_hot,
                                    binary=False)
    configs = None
    sentences = None
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (BCE + KLD) / x.size(0)

    def loss_fn_cont(recon_x, x, mean, log_var):
        MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (MSE + KLD) / x.size(0)
    return vocab, device, data_loader, loss_fn, inst_to_one_hot, \
           train_test_data, test_data, set_inds, states, s_min, s_max


def train(vocab, states, device, data_loader, loss_fn, inst_to_one_hot, train_test_data, test_data, set_inds,
          layers, embedding_size, latent_size, learning_rate, s_min, s_max, args):

    vae = ContextVAE(vocab.size, binary=False, inner_sizes=layers, state_size=states.shape[2], embedding_size=embedding_size, latent_size=latent_size).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    logs = defaultdict(list)
    env = gym.make('FetchManipulate3ObjectsContinuous-v0')
    get_config = env.unwrapped._get_configuration

    for epoch in range(args.epochs):

        for iteration, (init_config, sentence, config, init_state, state) in enumerate(data_loader):

            init_state, state, sentence = init_state.to(device), state.to(device), sentence.to(device)

            recon_state, mean, log_var, z = vae(init_state, sentence, state)

            loss = loss_fn(recon_state, state, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

        if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
            print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))


            score = 0
            score_dataset = 0
            inds = np.arange(len(train_test_data[0]))
            np.random.shuffle(inds)
            inds = inds[:1500]
            t_t_data = [np.array(train_test_data[i])[inds] for i in range(len(train_test_data))]
            for c_i, s, c_f_dataset, c_f_possible, co_i in zip(*t_t_data):
                one_hot = np.expand_dims(np.array(inst_to_one_hot[s.lower()]), 0)
                co_i = np.expand_dims(co_i, 0)
                co_i, s = torch.Tensor(co_i).to(device), torch.Tensor(one_hot).to(device)
                x = vae.inference(co_i, s, n=1).detach().numpy().flatten()#.astype(np.int)

                x = x * (s_max - s_min) + s_min

                x = get_config(x.reshape([3, 3])).astype(np.int)

                if str(x) in c_f_possible:
                    score += 1
                if str(x) in c_f_dataset:
                    score_dataset += 1
            print('Score train set: possible : {}, dataset : {}'.format(score / len(train_test_data[0]), score_dataset / len(train_test_data[0])))

    stop = 1

    results = np.zeros([len(set_inds), 2])
    # test train statistics
    factor = 50
    for i_gen in range(len(set_inds)):
        if i_gen == 0:
            set_name = 'Train'
        else:
            set_name = 'Test ' + str(i_gen)

        scores = []
        at_least_1 = []
        false_preds = []
        variabilities = []
        nb_cf_possible = []
        nb_cf_dataset = []
        if i_gen == 0:
            data_set = train_test_data
        else:
            data_set = test_data[i_gen - 1]
        for c_i, s, c_f_dataset, c_f_possible, co_i in zip(*data_set):
            one_hot = np.expand_dims(np.array(inst_to_one_hot[s.lower()]), 0)
            co_i = np.expand_dims(co_i, 0)
            one_hot = np.repeat(one_hot, factor, axis=0)
            co_i = np.repeat(co_i, factor, axis=0)
            co_i, s = torch.Tensor(co_i).to(device), torch.Tensor(one_hot).to(device)

            x = vae.inference(co_i, s, n=factor).detach().numpy()
            x_strs = []
            for xi in x:
                xi = get_config(xi.reshape([3, 3])).astype(np.int)
                x_strs.append(str(xi))
            variabilities.append(len(set(x_strs)))

            count_found = 0
            at_least_1_true = False
            false_preds.append(0)
            # count coverage of final configs in dataset
            for x_str in set(x_strs):
                if x_str in c_f_dataset:
                    count_found += 1
            # count false positives, final configs that are not compatible
            for x_str in x_strs:
                if x_str not in c_f_possible:
                    false_preds[-1] += 1
                else:
                    at_least_1_true = True
            scores.append(count_found / len(c_f_dataset))
            at_least_1.append(at_least_1_true)
            false_preds[-1] /= factor  # len(set(x_strs))
            nb_cf_possible.append(len(c_f_possible))
            nb_cf_dataset.append(len(c_f_dataset))
        print('\n{}: Average of percentage of final states found: {}'.format(set_name, np.mean(scores)))
        print('{}: At least one found: {}'.format(set_name, np.mean(at_least_1)))
        print('{}: Average variability: {}'.format(set_name, np.mean(variabilities)))
        print('{}: Average percentage of false preds: {}'.format(set_name, np.mean(false_preds)))
        print('{}: Average number of possible final configs: {}'.format(set_name, np.mean(nb_cf_possible)))
        print('{}: Average number of final configs in dataset: {}'.format(set_name, np.mean(nb_cf_dataset)))
        results[i_gen, 0] = np.mean(scores)
        results[i_gen, 1] = np.mean(false_preds)

    return results




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    # good ones
    embedding_size = 100
    layers = [128, 128]
    learning_rate = 0.005
    latent_size = 18

    vocab, device, data_loader, loss_fn_cont, inst_to_one_hot, \
    train_test_data, test_data, set_inds, states, s_min, s_max = main(args)

    train(vocab, states, device, data_loader, loss_fn_cont, inst_to_one_hot, train_test_data, test_data, set_inds,
          layers, embedding_size, latent_size, learning_rate,s_min, s_max,args)

    # import time
    # results = np.zeros([4, 3, 3, 3, 6, 2])
    # count = results.size / 12
    # counter = 0
    # path = '/home/flowers/Desktop/Scratch/sac_curriculum/language/data/'
    # for i, embedding_size in enumerate([10, 20, 50, 100]):
    #     for j, layers in enumerate([[64], [64, 64], [128, 128]]):
    #         for k, learning_rate in enumerate([0.001, 0.005, 0.01]):
    #             for l, latent_size in enumerate([9, 18, 27]):
    #                 t_i = time.time()
    #                 print('\n', embedding_size, layers, learning_rate, latent_size)
    #                 results[i, j, k, l, :, :] = train(vocab, configs, device, data_loader, loss_fn,
    #                                                   inst_to_one_hot, train_test_data, set_inds, sentences,
    #                                                   layers, embedding_size, latent_size, learning_rate, args)
    #                 with open(path + 'results.pk', 'wb') as f:
    #                     pickle.dump(results, f)
    #                 counter += 1
    #                 print(counter / count , '%', time.time() - t_i)


