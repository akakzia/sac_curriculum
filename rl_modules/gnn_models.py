import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import permutations, combinations
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class SinglePhiContext(nn.Module):
    def __init__(self, inp, out):
        super(SinglePhiContext, self).__init__()
        self.linear1 = nn.Linear(inp, 256)
        self.linear2 = nn.Linear(256, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = F.relu(self.linear1(inp))
        x = F.relu(self.linear2(x))

        return x


class RhoEncoder(nn.Module):
    def __init__(self, inp, out):
        super(RhoEncoder, self).__init__()
        self.linear1 = nn.Linear(inp, 256)
        self.linear2 = nn.Linear(256, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = self.linear1(inp)
        x = torch.sigmoid(self.linear2(x))

        return x


class SinglePhiActor(nn.Module):
    def __init__(self, inp, hid, out):
        super(SinglePhiActor, self).__init__()
        self.linear1 = nn.Linear(inp, hid)
        self.linear2 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = F.relu(self.linear1(inp))
        x = F.relu(self.linear2(x))

        return x


class RhoActor(nn.Module):
    def __init__(self, inp, out, action_space=None):
        super(RhoActor, self).__init__()
        self.linear1 = nn.Linear(inp, 256)
        self.mean_linear = nn.Linear(256, out)
        self.log_std_linear = nn.Linear(256, out)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class SinglePhiCritic(nn.Module):
    def __init__(self, inp, hid, out):
        super(SinglePhiCritic, self).__init__()
        self.linear1 = nn.Linear(inp, hid)
        self.linear2 = nn.Linear(hid, out)

        self.linear4 = nn.Linear(inp, hid)
        self.linear5 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x1 = F.relu(self.linear1(inp))
        x1 = F.relu(self.linear2(x1))

        x2 = F.relu(self.linear4(inp))
        x2 = F.relu(self.linear5(x2))

        return x1, x2


class RhoCritic(nn.Module):
    def __init__(self, inp, out):
        super(RhoCritic, self).__init__()
        self.linear1 = nn.Linear(inp, 256)
        self.linear3 = nn.Linear(256, out)

        self.linear4 = nn.Linear(inp, 256)
        self.linear6 = nn.Linear(256, out)

        self.apply(weights_init_)

    def forward(self, inp1, inp2):
        x1 = F.relu(self.linear1(inp1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(inp2))
        x2 = self.linear6(x2)

        return x1, x2


class DeepSetContext:
    def __init__(self, env_params, args):
        # A raw version of DeepSet-based SAC without attention mechanism
        self.observation = None
        self.ag = None
        self.g = None
        self.g_desc = None
        self.anchor_g = None
        self.latent = args.latent_dim
        self.dim_body = 10
        self.dim_object = 15
        self.dim_description = env_params['goal_size']
        self.dim_act = env_params['action']
        self.num_blocks = env_params['num_blocks']
        self.combinations_trick = args.combinations_trick
        self.aggregation = args.aggregation
        # if self.combinations_trick:
        #     self.n_permutations = len([x for x in combinations(range(self.num_blocks), 2)])
        # else:
        #     self.n_permutations = len([x for x in permutations(range(self.num_blocks), 2)])
        self.n_permutations = 2


        self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]

        self.context_tensor = None
        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        # dim_phi_encoder_input = self.dim_description[1]
        dim_phi_encoder_input = self.dim_description + 1
        dim_phi_encoder_output = 3 * dim_phi_encoder_input

        # dim_rho_encoder_input = dim_phi_encoder_output
        # dim_rho_encoder_output = self.latent

        # dim_input_objects = 2 * (self.num_blocks + self.dim_object)

        # dim_phi_actor_input = self.latent + self.dim_body + dim_input_objects
        dim_phi_actor_input = dim_phi_encoder_output + self.dim_body + self.dim_object
        dim_phi_actor_output = 3 * dim_phi_actor_input
        # dim_phi_actor_output = 3 * (self.dim_body + (self.num_blocks + self.dim_object) + self.latent)

        dim_rho_actor_input = dim_phi_actor_output
        dim_rho_actor_output = self.dim_act

        dim_phi_critic_input = dim_phi_encoder_output + self.dim_body + self.dim_object + self.dim_act
        dim_phi_critic_output = 3 * dim_phi_critic_input
        # dim_phi_critic_input = self.latent + self.dim_body + dim_input_objects + self.dim_act
        # dim_phi_critic_output = 3 * (self.dim_body + (self.num_blocks + self.dim_object) + self.dim_act + self.latent)

        dim_rho_critic_input = dim_phi_critic_output
        dim_rho_critic_output = 1

        self.single_phi_encoder = SinglePhiContext(dim_phi_encoder_input, dim_phi_encoder_output)
        # self.rho_encoder = RhoEncoder(dim_rho_encoder_input, dim_rho_encoder_output)

        self.single_phi_actor = SinglePhiActor(dim_phi_actor_input, 256, dim_phi_actor_output)
        self.rho_actor = RhoActor(dim_rho_actor_input, dim_rho_actor_output)

        self.single_phi_critic = SinglePhiCritic(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_critic = RhoCritic(dim_rho_critic_input, dim_rho_critic_output)

        self.single_phi_target_critic = SinglePhiCritic(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_target_critic = RhoCritic(dim_rho_critic_input, dim_rho_critic_output)

    def policy_forward_pass(self, obs, ag, g, no_noise=False):
        self.observation = obs
        self.ag = ag
        self.g = g

        obs_body = self.observation.narrow(-1, start=0, length=self.dim_body)
        # obs_objects = [torch.cat((torch.cat(obs_body.shape[0] * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.num_blocks),
        #                           self.observation.narrow(-1, start=self.dim_object*i + self.dim_body, length=self.dim_object)),
        #                          dim=-1) for i in range(self.num_blocks)]
        obs_objects = [self.ag.narrow(-1, start=2+self.dim_object*i, length=self.dim_object) for i in range(2)]

        # # Initialize context input
        # context_input = torch.empty((self.g_desc.shape[0], self.g_desc.shape[1], self.g_desc.shape[2] + 2*self.dim_object))
        #
        # # # Concatenate object observation to g description
        # for i, pair in enumerate(combinations(obs_objects, 2)):
        #     context_input[:, i, :] = torch.cat([self.g_desc[:, i, :5], pair[0][:, 3:], self.g_desc[:, i, 5:8], pair[1][:, 3:],
        #                                         self.g_desc[:, i, 8:]], dim=1)
        #
        # for i, pair in enumerate(permutations(obs_objects, 2)):
        #     context_input[:, i+3, :] = torch.cat([self.g_desc[:, i+3, :5], pair[0][:, 3:], self.g_desc[:, i+3, 5:8], pair[1][:, 3:],
        #                                           self.g_desc[:, i+3, 8:]], dim=1)
        #
        # output_phi_encoder = self.single_phi_encoder(context_input)
        edge_features_input = torch.cat((self.ag, self.g.narrow(-1, start=-1, length=1)), dim=-1)
        edge_features_output = self.single_phi_encoder(edge_features_input)

        # ids_edges = [np.array([0, 1, 5, 7]), np.array([0, 2, 3, 8]), np.array([1, 2, 4, 6])]

        # if self.aggregation == 'sum':
        #     input_actor = torch.stack([torch.cat([obs_body, obj, output_phi_encoder[:, ids_edges[i], :].sum(dim=1)], dim=1)
        #                                for i, obj in enumerate(obs_objects)])
        # else:
        #     input_actor = torch.stack([torch.cat([obs_body, obj, torch.max(output_phi_encoder[:, ids_edges[i], :], dim=1).values], dim=1)
        #                                for i, obj in enumerate(obs_objects)])

        input_actor = torch.stack([torch.cat([obs_body, obj, edge_features_output], dim=1) for obj in obs_objects])

        output_phi_actor = self.single_phi_actor(input_actor).sum(dim=0)

        # output_phi_encoder = self.single_phi_encoder(self.g_desc).sum(dim=1)
        #
        # self.context_tensor = self.rho_encoder(output_phi_encoder)
        #
        # if self.combinations_trick:
        #     # Get indexes of atomic goals and corresponding object tuple
        #     extractors = [torch.zeros((self.anchor_g.shape[1], 1)) for _ in range(self.anchor_g.shape[1])]
        #     for i in range(len(extractors)):
        #         extractors[i][i, :] = 1.
        #
        #     # The trick is to create selector matrices that, when multiplied with goals retrieves certain bits. Then the sign of the difference
        #     # between bits gives which objet goes above the the other
        #
        #     idxs_bits = [torch.empty(self.anchor_g.shape[0], 2) for _ in range(3)]
        #     idxs_objects = [torch.empty(self.anchor_g.shape[0], 2) for _ in range(3)]
        #
        #     for i, ((o1, o2), (j, k)) in enumerate(zip([(0, 1), (0, 2), (1, 2)], [(3, 5), (4, 7), (6, 8)])):
        #         stacked = torch.cat([extractors[j], extractors[k]], dim=1)
        #         multiplied_matrix = torch.matmul(self.anchor_g, stacked.double())
        #         selector = multiplied_matrix[:, 0] - multiplied_matrix[:, 1]
        #
        #         # idxs_objects[i] = torch.tensor([o2, o1]).repeat(self.anchor_g.shape[0], 1).long()
        #         # idxs_objects[i][selector >= 0] = torch.Tensor([o1, o2]).long()
        #         comb = list(permutations([o1, o2], 2))
        #         idxs_objects[i] = torch.stack([torch.tensor(comb[np.random.choice([0, 1])]) for _ in range(self.anchor_g.shape[0])]).long()
        #         idxs_objects[i][selector > 0] = torch.Tensor([o1, o2]).long()
        #         idxs_objects[i][selector < 0] = torch.Tensor([o2, o1]).long()
        #
        #     obs_object_tensor = torch.stack(obs_objects)
        #
        #     obs_objects_pairs_list = []
        #     for idxs_objects in idxs_objects:
        #         permuted_idxs = idxs_objects.unsqueeze(0).permute(2, 1, 0)
        #         permuted_idxs = permuted_idxs.repeat(1, 1, obs_object_tensor.shape[2])
        #         obs_objects_pair = obs_object_tensor.gather(0, permuted_idxs)
        #         obs_objects_pairs_list.append(obs_objects_pair)
        #
        #     input_actor = torch.stack([torch.cat([self.context_tensor, obs_body, obs_pair[0, :, :], obs_pair[1, :, :]], dim=1)
        #                                for obs_pair in obs_objects_pairs_list])
        #     # input_1_3 = torch.cat([ag_1_3, torch.cat([g_1_3, obs_body], dim=1), obs_objects_pairs_list[1][0, :, :],
        #     #                        obs_objects_pairs_list[1][1, :, :]], dim=1)
        #     # input_2_3 = torch.cat([ag_2_3, torch.cat([g_2_3, obs_body], dim=1), obs_objects_pairs_list[2][0, :, :],
        #     #                        obs_objects_pairs_list[2][1, :, :]], dim=1)
        #
        #     # input_actor = torch.stack([input_1_2, input_1_3, input_2_3])
        # else:
        #     input_actor = torch.stack([torch.cat([self.context_tensor, obs_body, x[0], x[1]], dim=1) for x in permutations(obs_objects, 2)])

        # self.save_values = self.single_phi_actor(input_actor).numpy()[:, 0, :]
        # output_phi_actor = self.single_phi_actor(input_actor).sum(dim=0)
        # self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)
        else:
            _, self.log_prob, self.pi_tensor = self.rho_actor.sample(output_phi_actor)

    def forward_pass(self, obs, ag, g, eval=False, actions=None):
        batch_size = obs.shape[0]
        self.observation = obs
        self.ag = ag
        self.g = g

        obs_body = self.observation[:, :self.dim_body]
        # obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.num_blocks),
        #                obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
        #                for i in range(self.num_blocks)]
        obs_objects = [self.ag.narrow(-1, start=2 + self.dim_object * i, length=self.dim_object) for i in range(2)]

        # output_phi_encoder = self.single_phi_encoder(self.g_desc).sum(dim=1)
        #
        # self.context_tensor = self.rho_encoder(output_phi_encoder)
        #
        # if self.combinations_trick:
        #     # Get indexes of atomic goals and corresponding object tuple
        #     extractors = [torch.zeros((self.anchor_g.shape[1], 1)) for _ in range(self.anchor_g.shape[1])]
        #     for i in range(len(extractors)):
        #         extractors[i][i, :] = 1.
        #
        #     # The trick is to create selector matrices that, when multiplied with goals retrieves certain bits. Then the sign of the difference
        #     # between bits gives which objet goes above the the other
        #
        #     idxs_bits = [torch.empty(self.anchor_g.shape[0], 2) for _ in range(3)]
        #     idxs_objects = [torch.empty(self.anchor_g.shape[0], 2) for _ in range(3)]
        #
        #     for i, ((o1, o2), (j, k)) in enumerate(zip([(0, 1), (0, 2), (1, 2)], [(3, 5), (4, 7), (6, 8)])):
        #         stacked = torch.cat([extractors[j], extractors[k]], dim=1)
        #         multiplied_matrix = torch.matmul(self.anchor_g, stacked.double())
        #         selector = multiplied_matrix[:, 0] - multiplied_matrix[:, 1]
        #
        #         # idxs_objects[i] = torch.tensor([o2, o1]).repeat(self.anchor_g.shape[0], 1).long()
        #         # idxs_objects[i][selector >= 0] = torch.Tensor([o1, o2]).long()
        #         comb = list(permutations([o1, o2], 2))
        #         idxs_objects[i] = torch.stack([torch.tensor(comb[np.random.choice([0, 1])]) for _ in range(self.anchor_g.shape[0])]).long()
        #         idxs_objects[i][selector > 0] = torch.Tensor([o1, o2]).long()
        #         idxs_objects[i][selector < 0] = torch.Tensor([o2, o1]).long()
        #
        #     obs_object_tensor = torch.stack(obs_objects)
        #
        #     obs_objects_pairs_list = []
        #     for idxs_objects in idxs_objects:
        #         permuted_idxs = idxs_objects.unsqueeze(0).permute(2, 1, 0)
        #         permuted_idxs = permuted_idxs.repeat(1, 1, obs_object_tensor.shape[2])
        #         obs_objects_pair = obs_object_tensor.gather(0, permuted_idxs)
        #         obs_objects_pairs_list.append(obs_objects_pair)
        #
        #     input_actor = torch.stack([torch.cat([self.context_tensor, obs_body, obs_pair[0, :, :], obs_pair[1, :, :]], dim=1)
        #                                for obs_pair in obs_objects_pairs_list])
        #
        # else:
        #     input_actor = torch.stack([torch.cat([self.context_tensor, obs_body, x[0], x[1]], dim=1) for x in permutations(obs_objects, 2)])
        #
        # output_phi_actor = self.single_phi_actor(input_actor).sum(dim=0)

        # # # Initialize context input
        # context_input = torch.empty((self.g_desc.shape[0], self.g_desc.shape[1], self.g_desc.shape[2] + 2 * self.dim_object))
        #
        # # # Concatenate object observation to g description
        # for i, pair in enumerate(combinations(obs_objects, 2)):
        #     context_input[:, i, :] = torch.cat([self.g_desc[:, i, :5], pair[0][:, 3:], self.g_desc[:, i, 5:8], pair[1][:, 3:],
        #                                         self.g_desc[:, i, 8:]], dim=1)
        #
        # for i, pair in enumerate(permutations(obs_objects, 2)):
        #     context_input[:, i + 3, :] = torch.cat([self.g_desc[:, i + 3, :5], pair[0][:, 3:], self.g_desc[:, i + 3, 5:8], pair[1][:, 3:],
        #                                             self.g_desc[:, i + 3, 8:]], dim=1)
        #
        # output_phi_encoder = self.single_phi_encoder(context_input)
        #
        # ids_edges = [np.array([0, 1, 5, 7]), np.array([0, 2, 3, 8]), np.array([1, 2, 4, 6])]
        #
        # if self.aggregation == 'sum':
        #     input_actor = torch.stack([torch.cat([obs_body, obj, output_phi_encoder[:, ids_edges[i], :].sum(dim=1)], dim=1)
        #                                for i, obj in enumerate(obs_objects)])
        # else:
        #     input_actor = torch.stack([torch.cat([obs_body, obj, output_phi_encoder[:, ids_edges[i], :].max(dim=1).values], dim=1)
        #                                for i, obj in enumerate(obs_objects)])
        #
        # output_phi_actor = self.single_phi_actor(input_actor).sum(dim=0)

        edge_features_input = torch.cat((self.ag, self.g.narrow(-1, start=-1, length=1)), dim=-1)
        edge_features_output = self.single_phi_encoder(edge_features_input)

        input_actor = torch.stack([torch.cat([obs_body, obj, edge_features_output], dim=1) for obj in obs_objects])

        output_phi_actor = self.single_phi_actor(input_actor).sum(dim=0)

        if not eval:
            self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)
        else:
            _, self.log_prob, self.pi_tensor = self.rho_actor.sample(output_phi_actor)

        # The critic part
        repeat_pol_actions = self.pi_tensor.repeat(self.n_permutations, 1, 1)
        input_critic = torch.cat([input_actor, repeat_pol_actions], dim=-1)
        if actions is not None:
            repeat_actions = actions.repeat(self.n_permutations, 1, 1)
            input_critic_with_act = torch.cat([input_actor, repeat_actions], dim=-1)
            input_critic = torch.cat([input_critic, input_critic_with_act], dim=0)

        with torch.no_grad():
            output_phi_target_critic_1, output_phi_target_critic_2 = self.single_phi_target_critic(input_critic[:self.n_permutations])
            output_phi_target_critic_1 = output_phi_target_critic_1.sum(dim=0)
            output_phi_target_critic_2 = output_phi_target_critic_2.sum(dim=0)
            self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.rho_target_critic(output_phi_target_critic_1, output_phi_target_critic_2)

        output_phi_critic_1, output_phi_critic_2 = self.single_phi_critic(input_critic)
        if actions is not None:
            output_phi_critic_1 = torch.stack([output_phi_critic_1[:self.n_permutations].sum(dim=0),
                                               output_phi_critic_1[self.n_permutations:].sum(dim=0)])
            output_phi_critic_2 = torch.stack([output_phi_critic_2[:self.n_permutations].sum(dim=0),
                                               output_phi_critic_2[self.n_permutations:].sum(dim=0)])
            q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
            self.q1_pi_tensor, self.q2_pi_tensor = q1_pi_tensor[0], q2_pi_tensor[0]
            return q1_pi_tensor[1], q2_pi_tensor[1]
        else:
            output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
            output_phi_critic_2 = output_phi_critic_2.sum(dim=0)
            self.q1_pi_tensor, self.q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
