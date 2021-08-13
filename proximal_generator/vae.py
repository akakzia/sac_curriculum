import torch
import torch.nn as nn
from itertools import combinations
from utils import get_per_object_predicate_indexes


def idx2onehot(idx, n):

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)

    return onehot


class ContextVAE(nn.Module):

    def __init__(self, encoder_inner_sizes=[32], decoder_inner_sizes=[32], state_size=9, latent_size=9,
                 binary=True, relational=False):
        super().__init__()

        assert type(encoder_inner_sizes) == list
        assert type(decoder_inner_sizes) == list
        assert type(latent_size) == int
        assert type(state_size) == int

        self.latent_size = latent_size
        self.state_size = state_size

        self.relational = relational
        self.o_ids = get_per_object_predicate_indexes(nb_objects=5)
        self.one_hot_encodings = [torch.tensor([1., 0., 0., 0., 0.]), torch.tensor([0., 1., 0., 0., 0.]),
                                  torch.tensor([0., 0., 1., 0., 0.]), torch.tensor([0., 0., 0., 1., 0.]),
                                  torch.tensor([0., 0., 0., 0., 1.])]
        # self.o_ids = [i for i in range(30)]

        encoder_layer_sizes = [5 + 15 + 12 * 2] + encoder_inner_sizes
        decoder_layer_sizes = [latent_size + 12 + 15 + 5] + decoder_inner_sizes + [12]
        self.encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes, binary=binary)

    def forward(self, initial_c, initial_s, current_c):

        batch_size = current_c.size(0)
        assert current_c.size(0) == initial_c.size(0) == initial_s.size(0)

        input_encoder = torch.stack([torch.cat((torch.cat(batch_size * [self.one_hot_encodings[k]]).reshape(batch_size, 5),
                                                initial_s, initial_c[:, o_ids],
                                                current_c[:, o_ids]), dim=1)
                                     for k, o_ids in enumerate(self.o_ids)])
        # means, log_var = self.encoder(torch.cat((initial_s, initial_c[:, self.o_ids], current_c[:, self.o_ids]), dim=1))
        means, log_var = self.encoder(input_encoder)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([5, batch_size, self.latent_size])
        z = eps * std + means

        input_decoder = torch.stack([torch.cat((torch.cat(batch_size * [self.one_hot_encodings[k]]).reshape(batch_size, 5),
                                                z[k], initial_s, initial_c[:, o_ids]), dim=1)
                                     for k, o_ids in enumerate(self.o_ids)])
        # recon_x = self.decoder(torch.cat((z, initial_s, initial_c[:, self.o_ids]), dim=1))
        recon_x = self.decoder(input_decoder)

        reconstructed = torch.zeros((batch_size, 30))
        for k, o_ids in enumerate(self.o_ids):
            reconstructed[:, o_ids] += recon_x[k]
        return reconstructed, means, log_var, z

    def inference(self, initial_s, initial_c, n=1):

        batch_size = n
        z = torch.randn([5, batch_size, self.latent_size])
        input_decoder = torch.stack([torch.cat((torch.cat(batch_size * [self.one_hot_encodings[k]]).reshape(batch_size, 5),
                                                z[k], initial_s, initial_c[:, o_ids]), dim=1)
                                     for k, o_ids in enumerate(self.o_ids)])
        # recon_state = self.decoder(torch.cat((z, initial_s, initial_c[:, self.o_ids]), dim=1))
        recon_state = self.decoder(input_decoder)

        res = torch.zeros((batch_size, 30))
        for k, o_ids in enumerate(self.o_ids):
            res[:, o_ids] += recon_state[k]
        return res


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, binary):

        super().__init__()

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 2 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                if binary:
                    self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):

        x = self.MLP(z)
        return x


class RelationalEncoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, initial_s, embeddings, current_s):
        batch_size = initial_s.shape[0]
        n_nodes = 3
        one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]
        ids_in_config = [[0, 3, 4], [1, 5, 6], [2, 7, 8]]

        inp = []
        count = 0
        for i, j in combinations([k for k in range(n_nodes)], 2):
            oh_i = torch.cat(batch_size * [one_hot_encodings[i]]).reshape(batch_size, n_nodes)
            oh_j = torch.cat(batch_size * [one_hot_encodings[j]]).reshape(batch_size, n_nodes)
            inp.append(torch.cat([oh_i, oh_j, initial_s[:, ids_in_config[count]], current_s[:, ids_in_config[count]], embeddings], dim=-1))
            count += 1

        inp = torch.stack(inp)
        x = self.MLP(inp)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class RelationalDecoder(nn.Module):

    def __init__(self, layer_sizes, binary):

        super().__init__()

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 2 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                if binary:
                    self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, embeddings, initial_s):
        batch_size = initial_s.shape[0]
        n_nodes = 3
        one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]
        ids_in_config = [[0, 3, 4], [1, 5, 6], [2, 7, 8]]

        inp = []
        count = 0
        for i, j in combinations([k for k in range(n_nodes)], 2):
            oh_i = torch.cat(batch_size * [one_hot_encodings[i]]).reshape(batch_size, n_nodes)
            oh_j = torch.cat(batch_size * [one_hot_encodings[j]]).reshape(batch_size, n_nodes)
            inp.append(torch.cat([oh_i, oh_j, initial_s[:, ids_in_config[count]], z[count], embeddings], dim=-1))
            count += 1

        inp = torch.stack(inp)
        x = self.MLP(inp)

        # flatten edges
        flat_x = torch.cat([x[i] for i in range(3)], dim=-1)[:, [0, 3, 4, 1, 5, 6, 2, 7, 8]]

        return flat_x