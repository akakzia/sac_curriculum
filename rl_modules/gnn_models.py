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


class EdgeEncoder(nn.Module):
    def __init__(self, inp, out):
        super(EdgeEncoder, self).__init__()
        self.linear1 = nn.Linear(inp, 256)
        self.linear2 = nn.Linear(256, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = F.relu(self.linear1(inp))
        x = F.relu(self.linear2(x))

        return x


class SinglePhiActor(nn.Module):
    def __init__(self, inp, hid, out):
        super(SinglePhiActor, self).__init__()
        self.linear1 = nn.Linear(inp, out)
        # self.linear2 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = F.relu(self.linear1(inp))
        # x = F.relu(self.linear2(x))

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
        self.linear1 = nn.Linear(inp, out)
        # self.linear2 = nn.Linear(hid, out)

        self.linear4 = nn.Linear(inp, out)
        # self.linear5 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x1 = F.relu(self.linear1(inp))
        # x1 = F.relu(self.linear2(x1))

        x2 = F.relu(self.linear4(inp))
        # x2 = F.relu(self.linear5(x2))

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


class GnnModel:
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
        self.dim_description = env_params['g_description']
        self.dim_act = env_params['action']
        self.num_blocks = env_params['num_blocks']
        self.combinations_trick = args.combinations_trick
        self.aggregation = args.aggregation

        self.context_tensor = None
        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        dim_edge_encoder_input = 4 + 2 * self.dim_object
        dim_edge_encoder_output = 3 * dim_edge_encoder_input

        dim_phi_actor_input = dim_edge_encoder_output + self.dim_body + self.dim_object
        dim_phi_actor_output = 3 * dim_phi_actor_input

        dim_rho_actor_input = dim_phi_actor_output
        dim_rho_actor_output = self.dim_act

        dim_phi_critic_input = dim_edge_encoder_output + self.dim_body + self.dim_object + self.dim_act
        dim_phi_critic_output = 3 * dim_phi_critic_input

        dim_rho_critic_input = dim_phi_critic_output
        dim_rho_critic_output = 1

        self.edge_encoder = EdgeEncoder(dim_edge_encoder_input, dim_edge_encoder_output)

        self.single_phi_actor = SinglePhiActor(dim_phi_actor_input, 256, dim_phi_actor_output)
        self.rho_actor = RhoActor(dim_rho_actor_input, dim_rho_actor_output)

        self.single_phi_critic = SinglePhiCritic(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_critic = RhoCritic(dim_rho_critic_input, dim_rho_critic_output)

        self.single_phi_target_critic = SinglePhiCritic(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_target_critic = RhoCritic(dim_rho_critic_input, dim_rho_critic_output)

    def policy_forward_pass(self, obs, g_desc, anchor_g=None, no_noise=False):
        self.observation = obs
        self.g_desc = g_desc

        obs_body = self.observation.narrow(-1, start=0, length=self.dim_body)
        obs_objects = [self.observation.narrow(-1, start=self.dim_object * i + self.dim_body, length=self.dim_object)
                       for i in range(self.num_blocks)]

        # # Initialize context input
        edge_inputs = torch.empty((self.g_desc.shape[0], self.g_desc.shape[1], 4 + 2*self.dim_object))

        # # Concatenate object observation to g description
        for i, pair in enumerate(combinations(obs_objects, 2)):
            edge_inputs[:, i, :] = torch.cat([self.g_desc[:, i, :2], pair[0], pair[1], self.g_desc[:, i, 2:]], dim=1)

        for i, pair in enumerate(permutations(obs_objects, 2)):
            edge_inputs[:, i+1, :] = torch.cat([self.g_desc[:, i+1, :2], pair[0], pair[1], self.g_desc[:, i+1, 2:]], dim=1)

        output_phi_encoder = self.edge_encoder(edge_inputs)

        ids_edges = [np.array([0, 2]), np.array([0, 1])]

        if self.aggregation == 'sum':
            input_actor = torch.stack([torch.cat([obs_body, obj, output_phi_encoder[:, ids_edges[i], :].sum(dim=1)], dim=1)
                                       for i, obj in enumerate(obs_objects)])
        else:
            input_actor = torch.stack([torch.cat([obs_body, obj, torch.max(output_phi_encoder[:, ids_edges[i], :], dim=1).values], dim=1)
                                       for i, obj in enumerate(obs_objects)])

        output_phi_actor = self.single_phi_actor(input_actor).sum(dim=0)

        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)
        else:
            _, self.log_prob, self.pi_tensor = self.rho_actor.sample(output_phi_actor)

    def forward_pass(self, obs, g_desc, anchor_g=None, eval=False, actions=None):
        self.observation = obs
        self.g_desc = g_desc
        self.anchor_g = anchor_g

        obs_body = self.observation[:, :self.dim_body]
        obs_objects = [self.observation.narrow(-1, start=self.dim_object * i + self.dim_body, length=self.dim_object)
                       for i in range(self.num_blocks)]

        # # Initialize context input
        edge_inputs = torch.empty((self.g_desc.shape[0], self.g_desc.shape[1], 4 + 2 * self.dim_object))

        # # Concatenate object observation to g description
        for i, pair in enumerate(combinations(obs_objects, 2)):
            edge_inputs[:, i, :] = torch.cat([self.g_desc[:, i, :2], pair[0], pair[1], self.g_desc[:, i, 2:]], dim=1)

        for i, pair in enumerate(permutations(obs_objects, 2)):
            edge_inputs[:, i + 1, :] = torch.cat([self.g_desc[:, i + 1, :2], pair[0], pair[1], self.g_desc[:, i + 1, 2:]], dim=1)

        output_phi_encoder = self.edge_encoder(edge_inputs)

        ids_edges = [np.array([0, 2]), np.array([0, 1])]

        if self.aggregation == 'sum':
            input_actor = torch.stack([torch.cat([obs_body, obj, output_phi_encoder[:, ids_edges[i], :].sum(dim=1)], dim=1)
                                       for i, obj in enumerate(obs_objects)])
        else:
            input_actor = torch.stack([torch.cat([obs_body, obj, output_phi_encoder[:, ids_edges[i], :].max(dim=1).values], dim=1)
                                       for i, obj in enumerate(obs_objects)])

        output_phi_actor = self.single_phi_actor(input_actor).sum(dim=0)

        if not eval:
            self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)
        else:
            _, self.log_prob, self.pi_tensor = self.rho_actor.sample(output_phi_actor)

        # The critic part
        repeat_pol_actions = self.pi_tensor.repeat(self.num_blocks, 1, 1)
        input_critic = torch.cat([input_actor, repeat_pol_actions], dim=-1)
        if actions is not None:
            repeat_actions = actions.repeat(self.num_blocks, 1, 1)
            input_critic_with_act = torch.cat([input_actor, repeat_actions], dim=-1)
            input_critic = torch.cat([input_critic, input_critic_with_act], dim=0)

        with torch.no_grad():
            output_phi_target_critic_1, output_phi_target_critic_2 = self.single_phi_target_critic(input_critic[:self.num_blocks])
            output_phi_target_critic_1 = output_phi_target_critic_1.sum(dim=0)
            output_phi_target_critic_2 = output_phi_target_critic_2.sum(dim=0)
            self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.rho_target_critic(output_phi_target_critic_1, output_phi_target_critic_2)

        output_phi_critic_1, output_phi_critic_2 = self.single_phi_critic(input_critic)
        if actions is not None:
            output_phi_critic_1 = torch.stack([output_phi_critic_1[:self.num_blocks].sum(dim=0),
                                               output_phi_critic_1[self.num_blocks:].sum(dim=0)])
            output_phi_critic_2 = torch.stack([output_phi_critic_2[:self.num_blocks].sum(dim=0),
                                               output_phi_critic_2[self.num_blocks:].sum(dim=0)])
            q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
            self.q1_pi_tensor, self.q2_pi_tensor = q1_pi_tensor[0], q2_pi_tensor[0]
            return q1_pi_tensor[1], q2_pi_tensor[1]
        else:
            output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
            output_phi_critic_2 = output_phi_critic_2.sum(dim=0)
            self.q1_pi_tensor, self.q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
