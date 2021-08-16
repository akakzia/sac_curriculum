import torch
import torch.nn as nn
from utils import get_graph_structure
import numpy as np


O_IDS = [[0, 1, 2, 3, 10, 11, 12, 13, 14, 18, 22, 26], [0, 4, 5, 6, 14, 15, 16, 17, 10, 19, 23, 27],
         [1, 4, 7, 8, 18, 19, 20, 21, 11, 15, 24, 28], [2, 5, 7, 9, 22, 23, 24, 25, 12, 16, 20, 29],
         [3, 6, 8, 9, 26, 27, 28, 29, 13, 17, 21, 25]]
# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ContextVAE(nn.Module):

    def __init__(self, encoder_inner_sizes=[32], decoder_inner_sizes=[32], state_size=9,
                 latent_size=9, binary=True, relational=False):
        super().__init__()

        assert type(encoder_inner_sizes) == list
        assert type(decoder_inner_sizes) == list
        assert type(latent_size) == int
        assert type(state_size) == int

        self.latent_size = latent_size
        self.state_size = state_size

        self.o_ids = [0, 1, 2, 3, 10, 11, 12, 13, 14, 18, 22, 26]

        self.relational = relational

        self.nb_objects = 5

        dim_mp_input = 5 * 2 + 3
        dim_mp_output = 3 * dim_mp_input
        self.mp_network = GnnMessagePassing(dim_mp_input, dim_mp_output)

        encoder_layer_sizes = [dim_mp_output] + encoder_inner_sizes
        decoder_layer_sizes = [latent_size + 5 + 12] + decoder_inner_sizes + [12] # 12 = 4 * 3
        self.encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes, binary=binary)

        # self.edges, self.incoming_edges, self.predicate_ids = get_graph_structure(self.nb_objects)
        self.edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
        self.predicate_ids = [[0, 10, 14], [1, 11, 18], [2, 12, 22], [3, 13, 26]]
        self.incoming_edges = [[0, 1, 2, 3], [], [], [], []]

        self.nb_permutations = len(self.edges)

        self.one_hot_encodings = [torch.tensor([1., 0., 0., 0., 0.]), torch.tensor([0., 1., 0., 0., 0.]),
                                  torch.tensor([0., 0., 1., 0., 0.]), torch.tensor([0., 0., 0., 1., 0.]),
                                  torch.tensor([0., 0., 0., 0., 1.])]

    def forward(self, initial_c, initial_s, current_c):
        batch_size = current_c.size(0)
        assert current_c.size(0) == initial_s.size(0) == initial_c.size(0)

        # nodes = torch.cat([torch.cat(batch_size * self.one_hot_encodings).reshape(batch_size, self.nb_objects, self.nb_objects),
        #                    initial_s.reshape((batch_size, self.nb_objects, -1))], dim=-1)

        # nodes = initial_s.reshape((batch_size, self.nb_objects, -1))

        nodes = torch.cat(batch_size * self.one_hot_encodings).reshape(batch_size, self.nb_objects, self.nb_objects)

        # Message Passing Step to compute updated edge features
        inp_mp = torch.stack([torch.cat([nodes[:, self.edges[i][0], :], nodes[:, self.edges[i][1], :],
                                         current_c[:, self.predicate_ids[i]]], dim=-1) for i in range(self.nb_permutations)])

        means, log_var = self.mp_network(inp_mp)

        # Node update Step (Encoder)
        # inp_encoder = torch.stack([torch.cat([nodes[:, i, :], torch.sum(edge_features[self.incoming_edges[i], :, :], dim=0)], dim=1)
        #                    for i in range(self.nb_objects)])[0]
        # inp_encoder = torch.cat([nodes[:, 0, :], torch.sum(edge_features[self.incoming_edges[0], :, :], dim=0)], dim=1)
        # inp_encoder = torch.sum(edge_features[self.incoming_edges[0], :, :], dim=0)
        # means, log_var = self.encoder(inp_encoder)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * std + means

        # Decoder step
        # inp_decoder = torch.stack([torch.cat([nodes[:, i, :], z[i]], dim=1)
        #                    for i in range(self.nb_objects)])
        inp_decoder = torch.cat([nodes[:, 0, :], initial_c[:, self.o_ids], z], dim=1)

        recon_x = self.decoder(inp_decoder)

        return recon_x, means, log_var, z

    def inference(self, initial_s, initial_c, n=1, index=[0]):
        batch_size = n

        # nodes = torch.cat([torch.cat(batch_size * self.one_hot_encodings).reshape(batch_size, self.nb_objects, self.nb_objects),
        #                    initial_s.reshape((batch_size, self.nb_objects, -1))], dim=-1)
        nodes = torch.cat(batch_size * self.one_hot_encodings).reshape(batch_size, self.nb_objects, self.nb_objects)
        # nodes = initial_s.reshape((batch_size, self.nb_objects, -1))
        # z = torch.cat([torch.randn([1, batch_size, self.latent_size]), torch.zeros([self.nb_objects-1, batch_size, self.latent_size])])
        z = torch.randn([batch_size, self.latent_size])
        # inp_decoder = torch.stack([torch.cat([nodes[:, i, :], z[i]], dim=1)
        #                            for i in range(self.nb_objects)])
        # inp_decoder = torch.cat([nodes[:, 0, :], initial_c[:, O_IDS[0]], z], dim=1)
        inp_decoder = torch.stack([torch.cat([nodes[i, 0, :], initial_c[i, O_IDS[ie]], z[i, :]]) for i, ie in enumerate(index)])

        recon_state = self.decoder(inp_decoder)

        res = torch.clone(initial_c)
        for i in range(batch_size):
            res[i, O_IDS[index[i]]] = recon_state[i, :]

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

        self.apply(weights_init_)
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
        self.apply(weights_init_)

    def forward(self, z):

        x = self.MLP(z)
        return x


class GnnMessagePassing(nn.Module):
    def __init__(self, inp, out):
        super(GnnMessagePassing, self).__init__()
        self.linear1 = nn.Linear(inp, 128)
        self.linear2 = nn.Linear(128, out)

        self.linear3 = nn.Linear(out, 128)

        self.linear_means = nn.Linear(128, 8)
        self.linear_log_var = nn.Linear(128, 8)

        self.apply(weights_init_)

    def forward(self, inp):
        x = nn.functional.relu(self.linear1(inp))
        x = nn.functional.relu(self.linear2(x))

        # Aggregation
        x = x.sum(0)

        x = nn.functional.relu(self.linear3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars