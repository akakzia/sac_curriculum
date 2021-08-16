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
        # self.o_ids = get_per_object_predicate_indexes(nb_objects=5)
        self.o_ids = [0, 1, 2, 3, 10, 11, 12, 13, 14, 18, 22, 26]
        self.one_hot_encodings = [torch.tensor([1., 0., 0., 0., 0.]), torch.tensor([0., 1., 0., 0., 0.]),
                                  torch.tensor([0., 0., 1., 0., 0.]), torch.tensor([0., 0., 0., 1., 0.]),
                                  torch.tensor([0., 0., 0., 0., 1.])]

        encoder_layer_sizes = [15 + 12 * 2] + encoder_inner_sizes
        decoder_layer_sizes = [latent_size + 15 + 12] + decoder_inner_sizes + [12]
        self.encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes, binary=binary)

    def forward(self, initial_c, initial_s, current_c):

        batch_size = current_c.size(0)
        assert current_c.size(0) == initial_c.size(0) == initial_s.size(0)

        input_encoder = torch.cat((initial_s, initial_c[:, self.o_ids], current_c[:, self.o_ids]), dim=1)
        means, log_var = self.encoder(input_encoder)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * std + means

        decoder_input = torch.cat((z, initial_s, initial_c[:, self.o_ids]), dim=1)
        recon_x = self.decoder(decoder_input)

        # res = torch.clone(initial_c)
        # res[:, self.o_ids] = recon_x

        return recon_x, means, log_var, z

    def inference(self, initial_s, initial_c, n=1):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size])
        # input_decoder = torch.stack([torch.cat((torch.cat(batch_size * [self.one_hot_encodings[k]]).reshape(batch_size, 5),
        #                                         z[k], initial_s, initial_c[:, o_ids]), dim=1)
        #                              for k, o_ids in enumerate(self.o_ids)])
        input_decoder = torch.cat((z, initial_s, initial_c[:, self.o_ids]), dim=1)
        recon_state = self.decoder(input_decoder)

        # res = torch.clone(initial_c)
        # res[:, self.o_ids] = recon_state

        return recon_state


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