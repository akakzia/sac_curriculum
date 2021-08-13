import torch
# from vae import ContextVAE
from gvae import ContextVAE
from collections import defaultdict
import numpy as np
from graph.teacher import Teacher


class ProximalGoalGenerator:
    def __init__(self, initial_configurations, data_loader,  device, args):
        self.encoder_layers = args.encoder_layer_sizes
        self.decoder_layers = args.decoder_layer_sizes
        self.state_size = initial_configurations.shape[-1]
        self.latent_size = args.latent_size
        self.data_loader = data_loader
        self.device = device

        self.vae = ContextVAE(encoder_inner_sizes=self.encoder_layers, decoder_inner_sizes=self.decoder_layers,
                              state_size=self.state_size, latent_size=self.latent_size,
                              relational=args.relational).to(self.device)
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=args.learning_rate)

        def loss_fn(recon_x, x, mean, log_var):
            bce = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
            kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            return (bce + args.k_param * kld) / x.size(0)

        self.loss_fn = loss_fn

        self.save_path = args.save_path

        self.teacher = Teacher(args)

    def train(self):
        logs = defaultdict(list)

        for iteration, (init_config, init_state, final_config) in enumerate(self.data_loader):
            init_config, init_state, final_config = init_config.to(self.device), init_state.to(self.device),\
                                                    final_config.to(self.device)

            recon_state, mean, log_var, z = self.vae(init_config, init_state, final_config)

            target = final_config
            loss = self.loss_fn(recon_state, target, mean, log_var)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            logs['loss'].append(loss.item())
        print('Updates = {} | Loss = {}'.format(iteration, loss.item()))

    def save(self, vae_id):
        with open(self.save_path + 'vae_model{}.pkl'.format(vae_id), 'wb') as f:
            torch.save(self.vae, f)

    def evaluate(self, init_configs, init_states, final_configs, set_inds, init_to_finals):
        results = np.zeros([len(set_inds), 3])

        # test train statistics
        factor = 100
        for i_gen in range(len(set_inds)):
            if i_gen == 0:
                set_name = 'Train'
            else:
                set_name = 'Test ' + str(i_gen)

            coverage_dataset = []
            coverage_possible = []
            count = 0
            false_preds = []
            variabilities = []
            valid_goals = []
            data_set = (init_configs[set_inds[i_gen][:100]], init_states[set_inds[i_gen][:100]],
                        final_configs[set_inds[i_gen]][:100])
            for c_i, s_i, c_f in zip(*data_set):
                count += 1
                c_ii = np.expand_dims(c_i, 0)
                c_ii = np.repeat(c_ii, factor, axis=0)
                c_ii = torch.Tensor(c_ii).to(self.device)
                s_ii = np.expand_dims(s_i, 0)
                s_ii = np.repeat(s_ii, factor, axis=0)
                s_ii = torch.Tensor(s_ii).to(self.device)

                neighbours = self.get_neighbours(c_i)
                x = (self.vae.inference(s_ii, c_ii, n=factor).detach().numpy() > 0.5).astype(np.float32)

                x_strs = [str(xi) for xi in x]
                variabilities.append(len(set(x_strs)))
                count_found_possible = 0
                count_false_pred = 0

                for x_str in set(x_strs):
                    # if x_str in init_to_finals[str(c_i)]:
                    if x_str in neighbours:
                        count_found_possible += 1
                    else:
                        stop = 1

                # count false positives, final configs that are not compatible
                for x_str in x_strs:
                    # if x_str not in init_to_finals[str(c_i)]:
                    if x_str not in neighbours:
                        count_false_pred += 1

                coverage_possible.append(count_found_possible / max(len(neighbours), 1))
                valid_goals.append(count_found_possible)
                false_preds.append(count_false_pred / factor)
            print('\n{}: Probability that a sampled goal is valid {}'.format(set_name, 1 - np.mean(false_preds)))
            print('{}: Number of different valid sampled goals: {}'.format(set_name, np.mean(valid_goals)))
            print('{}: Coverage of all valid goals: {}'.format(set_name, np.mean(coverage_possible)))
            results[i_gen, 0] = count
            results[i_gen, 1] = 1 - np.mean(false_preds)
            results[i_gen, 2] = np.mean(valid_goals)

        return results

    def get_neighbours(self, configuration):
        """ Given a semantic configuration returns the set of neighbors using the oracle graph"""
        c = configuration.copy()
        c[np.where(configuration == 0.)] = -1
        neighbours = list(filter( lambda x : x not in [], self.teacher.oracle_graph.iterNeighbors(tuple(c))))
        neighbours = set([str(np.clip(np.array(n) + 1, 0, 1)) for n in neighbours])
        return neighbours