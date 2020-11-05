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


class MessagePassing(nn.Module):
    def __init__(self, inp, out):
        super(MessagePassing, self).__init__()
        self.linear1 = nn.Linear(inp, 256)
        self.linear2 = nn.Linear(256, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = F.relu(self.linear1(inp))
        x = F.relu(self.linear2(x))

        return x


class NodeAggregation(nn.Module):
    def __init__(self, inp, hid, out):
        super(NodeAggregation, self).__init__()
        self.linear1 = nn.Linear(inp, hid)
        self.linear2 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = F.relu(self.linear1(inp))
        x = F.relu(self.linear2(x))

        return x


class GraphAggregation(nn.Module):
    def __init__(self, inp, hid, out):
        super(GraphAggregation, self).__init__()
        self.linear1 = nn.Linear(inp, hid)
        self.linear2 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = F.relu(self.linear1(inp))
        x = F.relu(self.linear2(x))

        return x


class GnnStateEncoder:
    def __init__(self, dim_edge, dim_node, dim_latent, aggr='sum'):
        self.aggr = aggr

        input_mp = dim_edge
        output_mp = 3 * input_mp
        self.message_passing_module = MessagePassing(input_mp, output_mp)

        input_node_aggr = output_mp + dim_node
        output_node_aggr = 3 * input_node_aggr
        self.node_aggr_module = NodeAggregation(input_node_aggr, 256, output_node_aggr)

        input_graph_aggr = output_node_aggr
        output_graph_aggr = dim_latent
        self.graph_aggr_module = GraphAggregation(input_graph_aggr, 256, output_graph_aggr)

    def propagate(self, inp):
        return self.message_passing_module(inp)

    def node_aggr(self, obs_objects, updated_edges, ids_edges):
        if self.aggr == 'sum':
            inp = torch.stack([torch.cat([obj, updated_edges[:, ids_edges[i], :].sum(dim=1)], dim=1)
                                       for i, obj in enumerate(obs_objects)])
        else:
            inp = torch.stack([torch.cat([obj, torch.max(updated_edges[:, ids_edges[i], :], dim=1).values], dim=1)
                                       for i, obj in enumerate(obs_objects)])

        return self.node_aggr_module(inp).sum(dim=0)

    def graph_aggr(self, inp):
        return self.graph_aggr_module(inp)


class ActorNet(nn.Module):
    def __init__(self, inp, out, action_space=None):
        super(ActorNet, self).__init__()
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


class CriticNet(nn.Module):
    def __init__(self, inp, out):
        super(CriticNet, self).__init__()
        self.linear1 = nn.Linear(inp, 256)
        self.linear3 = nn.Linear(256, out)

        self.linear4 = nn.Linear(inp, 256)
        self.linear6 = nn.Linear(256, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x1 = F.relu(self.linear1(inp))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(inp))
        x2 = self.linear6(x2)

        return x1, x2


# class SinglePhiContext(nn.Module):
#     def __init__(self, inp, out):
#         super(SinglePhiContext, self).__init__()
#         self.linear1 = nn.Linear(inp, 256)
#         self.linear2 = nn.Linear(256, out)
#
#         self.apply(weights_init_)
#
#     def forward(self, inp):
#         x = F.relu(self.linear1(inp))
#         x = F.relu(self.linear2(x))
#
#         return x
#
#
# class SinglePhiActor(nn.Module):
#     def __init__(self, inp, hid, out):
#         super(SinglePhiActor, self).__init__()
#         self.linear1 = nn.Linear(inp, hid)
#         self.linear2 = nn.Linear(hid, out)
#
#         self.apply(weights_init_)
#
#     def forward(self, inp):
#         x = F.relu(self.linear1(inp))
#         x = F.relu(self.linear2(x))
#
#         return x
#
#
# class RhoActor(nn.Module):
#     def __init__(self, inp, out, action_space=None):
#         super(RhoActor, self).__init__()
#         self.linear1 = nn.Linear(inp, 256)
#         self.mean_linear = nn.Linear(256, out)
#         self.log_std_linear = nn.Linear(256, out)
#
#         self.apply(weights_init_)
#
#         # action rescaling
#         if action_space is None:
#             self.action_scale = torch.tensor(1.)
#             self.action_bias = torch.tensor(0.)
#         else:
#             self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
#             self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)
#
#     def forward(self, x):
#         x = F.relu(self.linear1(x))
#         mean = self.mean_linear(x)
#         log_std = self.log_std_linear(x)
#         log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
#         return mean, log_std
#
#     def sample(self, state):
#         mean, log_std = self.forward(state)
#         std = log_std.exp()
#         normal = Normal(mean, std)
#         x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         y_t = torch.tanh(x_t)
#         action = y_t * self.action_scale + self.action_bias
#         log_prob = normal.log_prob(x_t)
#         # Enforcing Action Bound
#         log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
#         log_prob = log_prob.sum(-1, keepdim=True)
#         return action, log_prob, torch.tanh(mean)
#
#
# class SinglePhiCritic(nn.Module):
#     def __init__(self, inp, hid, out):
#         super(SinglePhiCritic, self).__init__()
#         self.linear1 = nn.Linear(inp, hid)
#         self.linear2 = nn.Linear(hid, out)
#
#         self.linear4 = nn.Linear(inp, hid)
#         self.linear5 = nn.Linear(hid, out)
#
#         self.apply(weights_init_)
#
#     def forward(self, inp):
#         x1 = F.relu(self.linear1(inp))
#         x1 = F.relu(self.linear2(x1))
#
#         x2 = F.relu(self.linear4(inp))
#         x2 = F.relu(self.linear5(x2))
#
#         return x1, x2
#
#
# class RhoCritic(nn.Module):
#     def __init__(self, inp, out):
#         super(RhoCritic, self).__init__()
#         self.linear1 = nn.Linear(inp, 256)
#         self.linear3 = nn.Linear(256, out)
#
#         self.linear4 = nn.Linear(inp, 256)
#         self.linear6 = nn.Linear(256, out)
#
#         self.apply(weights_init_)
#
#     def forward(self, inp1, inp2):
#         x1 = F.relu(self.linear1(inp1))
#         x1 = self.linear3(x1)
#
#         x2 = F.relu(self.linear4(inp2))
#         x2 = self.linear6(x2)
#
#         return x1, x2


class GNN:
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
        if self.combinations_trick:
            self.n_permutations = len([x for x in combinations(range(self.num_blocks), 2)])
        else:
            self.n_permutations = len([x for x in permutations(range(self.num_blocks), 2)])

        self.one_hot_encodings = [torch.tensor([1., 0.]), torch.tensor([0., 1.])]

        self.context_tensor = None
        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        # dim_phi_encoder_input = self.dim_description[1]
        # dim_phi_encoder_input = 3 + 2 * self.dim_object
        # dim_phi_encoder_output = 3 * dim_phi_encoder_input

        # dim_input_objects = 2 * (self.num_blocks + self.dim_object)

        # dim_phi_actor_input = self.latent + self.dim_body + dim_input_objects
        # dim_phi_actor_input = dim_phi_encoder_output + self.dim_object
        # dim_phi_actor_output = 3 * dim_phi_actor_input
        # dim_phi_actor_output = 3 * (self.dim_body + (self.num_blocks + self.dim_object) + self.latent)

        # dim_rho_actor_input = dim_phi_actor_output + self.dim_body
        # dim_rho_actor_output = self.dim_act

        # dim_phi_critic_input = dim_phi_encoder_output + self.dim_object
        # dim_phi_critic_output = 3 * dim_phi_critic_input
        # dim_phi_critic_input = self.latent + self.dim_body + dim_input_objects + self.dim_act
        # dim_phi_critic_output = 3 * (self.dim_body + (self.num_blocks + self.dim_object) + self.dim_act + self.latent)

        # dim_rho_critic_input = self.dim_body + self.dim_act + dim_phi_actor_output
        # dim_rho_critic_output = 1

        # self.single_phi_encoder = SinglePhiContext(dim_phi_encoder_input, dim_phi_encoder_output)
        #
        # self.single_phi_actor = SinglePhiActor(dim_phi_actor_input, 256, dim_phi_actor_output)
        # self.rho_actor = RhoActor(dim_rho_actor_input, dim_rho_actor_output)
        #
        # self.single_phi_critic = SinglePhiCritic(dim_phi_critic_input, 256, dim_phi_critic_output)
        # self.rho_critic = RhoCritic(dim_rho_critic_input, dim_rho_critic_output)
        #
        # self.single_phi_target_critic = SinglePhiCritic(dim_phi_critic_input, 256, dim_phi_critic_output)
        # self.rho_target_critic = RhoCritic(dim_rho_critic_input, dim_rho_critic_output)

        dim_phi_encoder_input = 3 + 2 * self.dim_object
        self.current_state_encoder = GnnStateEncoder(dim_phi_encoder_input, self.dim_object, 10, aggr=self.aggregation)
        self.target_state_encoder = GnnStateEncoder(dim_phi_encoder_input, self.dim_object, 10, aggr=self.aggregation)

        dim_actor_input = self.dim_body + self.num_blocks * self.dim_object + 2 * 10
        self.actor_network = ActorNet(dim_actor_input, self.dim_act)

        dim_critic_input = dim_actor_input + self.dim_act
        self.critic_network = CriticNet(dim_critic_input, 1)
        self.target_critic_network = CriticNet(dim_critic_input, 1)

    def policy_forward_pass(self, obs, g_desc, anchor_g=None, no_noise=False):
        self.observation = obs
        self.g_desc = g_desc
        self.anchor_g = anchor_g

        obs_body = self.observation.narrow(-1, start=0, length=self.dim_body)
        obs_objects = [self.observation.narrow(-1, start=self.dim_object * i + self.dim_body, length=self.dim_object)
                       for i in range(self.num_blocks)]

        # # Initialize context input
        message_input_current = torch.empty((self.g_desc.shape[0], self.g_desc.shape[1], 3 + 2*self.dim_object))
        message_input_target = torch.empty((self.g_desc.shape[0], self.g_desc.shape[1], 3 + 2 * self.dim_object))

        # # Concatenate object observation to g description
        for i, pair in enumerate(combinations(obs_objects, 2)):
            message_input_current[:, i, :] = torch.cat([self.g_desc[:, i, :2], pair[0], pair[1], self.g_desc[:, i, 6:7]], dim=1)
            message_input_target[:, i, :] = torch.cat([self.g_desc[:, i, :2], pair[0], pair[1], self.g_desc[:, i, 7:8]], dim=1)

        for i, pair in enumerate(permutations(obs_objects, 2)):
            message_input_current[:, i+1, :] = torch.cat([self.g_desc[:, i+1, :2], pair[0], pair[1], self.g_desc[:, i+1, 6:7]], dim=1)
            message_input_target[:, i + 1, :] = torch.cat([self.g_desc[:, i + 1, :2], pair[0], pair[1], self.g_desc[:, i + 1, 7:8]], dim=1)

        ids_edges = [np.array([0, 2]), np.array([0, 1])]

        updated_edges = self.current_state_encoder.propagate(message_input_current)
        aggregated_nodes_emb = self.current_state_encoder.node_aggr(obs_objects, updated_edges, ids_edges)
        current_state_embedding = self.current_state_encoder.graph_aggr(aggregated_nodes_emb)

        target_updated_edges = self.target_state_encoder.propagate(message_input_target)
        target_aggregated_nodes_emb = self.target_state_encoder.node_aggr(obs_objects, target_updated_edges, ids_edges)
        target_state_embedding = self.target_state_encoder.graph_aggr(target_aggregated_nodes_emb)

        input_actor = torch.cat([self.observation, current_state_embedding, target_state_embedding], dim=-1)

        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.actor_network.sample(input_actor)
        else:
            _, self.log_prob, self.pi_tensor = self.actor_network.sample(input_actor)

        # output_phi_encoder = self.single_phi_encoder(context_input)
        #
        # ids_edges = [np.array([0, 2]), np.array([0, 1])]
        #
        # if self.aggregation == 'sum':
        #     input_actor = torch.stack([torch.cat([obj, output_phi_encoder[:, ids_edges[i], :].sum(dim=1)], dim=1)
        #                                for i, obj in enumerate(obs_objects)])
        # else:
        #     input_actor = torch.stack([torch.cat([obj, torch.max(output_phi_encoder[:, ids_edges[i], :], dim=1).values], dim=1)
        #                                for i, obj in enumerate(obs_objects)])
        #
        # output_phi_actor = self.single_phi_actor(input_actor).sum(dim=0)
        #
        # input_rho_actor = torch.cat([obs_body, output_phi_actor], dim=1)

        # if not no_noise:
        #     self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(input_rho_actor)
        # else:
        #     _, self.log_prob, self.pi_tensor = self.rho_actor.sample(input_rho_actor)

    def forward_pass(self, obs, g_desc, anchor_g=None, eval=False, actions=None):
        self.observation = obs
        self.g_desc = g_desc
        self.anchor_g = anchor_g

        obs_body = self.observation[:, :self.dim_body]
        obs_objects = [self.observation.narrow(-1, start=self.dim_object * i + self.dim_body, length=self.dim_object)
                       for i in range(self.num_blocks)]

        # # # Initialize context input
        # context_input = torch.empty((self.g_desc.shape[0], self.g_desc.shape[1], 4 + 2 * self.dim_object))
        #
        # # # Concatenate object observation to g description
        # for i, pair in enumerate(combinations(obs_objects, 2)):
        #     context_input[:, i, :] = torch.cat([self.g_desc[:, i, :2], pair[0], pair[1], self.g_desc[:, i, 6:]], dim=1)
        #
        # for i, pair in enumerate(permutations(obs_objects, 2)):
        #     context_input[:, i + 1, :] = torch.cat([self.g_desc[:, i + 1, :2], pair[0], pair[1], self.g_desc[:, i + 1, 6:]], dim=1)
        #
        # output_phi_encoder = self.single_phi_encoder(context_input)
        #
        # ids_edges = [np.array([0, 2]), np.array([0, 1])]
        #
        # if self.aggregation == 'sum':
        #     input_actor = torch.stack([torch.cat([obj, output_phi_encoder[:, ids_edges[i], :].sum(dim=1)], dim=1)
        #                                for i, obj in enumerate(obs_objects)])
        # else:
        #     input_actor = torch.stack([torch.cat([obj, output_phi_encoder[:, ids_edges[i], :].max(dim=1).values], dim=1)
        #                                for i, obj in enumerate(obs_objects)])
        #
        # output_phi_actor = self.single_phi_actor(input_actor).sum(dim=0)
        #
        # input_rho_actor = torch.cat([obs_body, output_phi_actor], dim=1)
        #
        # if not eval:
        #     self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(input_rho_actor)
        # else:
        #     _, self.log_prob, self.pi_tensor = self.rho_actor.sample(input_rho_actor)

        # # Initialize context input
        message_input_current = torch.empty((self.g_desc.shape[0], self.g_desc.shape[1], 3 + 2 * self.dim_object))
        message_input_target = torch.empty((self.g_desc.shape[0], self.g_desc.shape[1], 3 + 2 * self.dim_object))

        # # Concatenate object observation to g description
        for i, pair in enumerate(combinations(obs_objects, 2)):
            message_input_current[:, i, :] = torch.cat([self.g_desc[:, i, :2], pair[0], pair[1], self.g_desc[:, i, 6:7]], dim=1)
            message_input_target[:, i, :] = torch.cat([self.g_desc[:, i, :2], pair[0], pair[1], self.g_desc[:, i, 7:8]], dim=1)

        for i, pair in enumerate(permutations(obs_objects, 2)):
            message_input_current[:, i + 1, :] = torch.cat([self.g_desc[:, i + 1, :2], pair[0], pair[1], self.g_desc[:, i + 1, 6:7]], dim=1)
            message_input_target[:, i + 1, :] = torch.cat([self.g_desc[:, i + 1, :2], pair[0], pair[1], self.g_desc[:, i + 1, 7:8]], dim=1)

        ids_edges = [np.array([0, 2]), np.array([0, 1])]

        updated_edges = self.current_state_encoder.propagate(message_input_current)
        aggregated_nodes_emb = self.current_state_encoder.node_aggr(obs_objects, updated_edges, ids_edges)
        current_state_embedding = self.current_state_encoder.graph_aggr(aggregated_nodes_emb)

        target_updated_edges = self.target_state_encoder.propagate(message_input_target)
        target_aggregated_nodes_emb = self.target_state_encoder.node_aggr(obs_objects, target_updated_edges, ids_edges)
        target_state_embedding = self.target_state_encoder.graph_aggr(target_aggregated_nodes_emb)

        input_actor = torch.cat([self.observation, current_state_embedding, target_state_embedding], dim=-1)

        if not eval:
            self.pi_tensor, self.log_prob, _ = self.actor_network.sample(input_actor)
        else:
            _, self.log_prob, self.pi_tensor = self.actor_network.sample(input_actor)

        input_critic = torch.cat([self.observation, self.pi_tensor, current_state_embedding, target_state_embedding], dim=-1)
        if actions is not None:
            input_critic_with_act = torch.cat([self.observation, actions, current_state_embedding, target_state_embedding], dim=-1)
            input_critic = torch.stack([input_critic, input_critic_with_act])

        with torch.no_grad():
            self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.target_critic_network(input_critic)

        if actions is not None:
            q1_pi_tensor, q2_pi_tensor = self.critic_network(input_critic)
            self.q1_pi_tensor, self.q2_pi_tensor = q1_pi_tensor[0], q2_pi_tensor[0]
            return q1_pi_tensor[1], q2_pi_tensor[1]

        self.q1_pi_tensor, self.q2_pi_tensor = self.critic_network(input_critic)

        # The critic part
        # repeat_pol_actions = self.pi_tensor.repeat(self.num_blocks, 1, 1)
        # output_phi_critic_1, output_phi_critic_2 = self.single_phi_critic(input_actor)
        # output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
        # output_phi_critic_2 = output_phi_critic_2.sum(dim=0)
        # input_rho_critic_1 = torch.cat([obs_body, self.pi_tensor, output_phi_critic_1], dim=-1)
        # input_rho_critic_2 = torch.cat([obs_body, self.pi_tensor, output_phi_critic_2], dim=-1)
        # if actions is not None:
        #     # repeat_actions = actions.repeat(self.num_blocks, 1, 1)
        #     input_rho_critic_with_act_1 = torch.cat([obs_body, actions, output_phi_critic_1], dim=-1)
        #     input_rho_critic_with_act_2 = torch.cat([obs_body, actions, output_phi_critic_2], dim=-1)
        #     input_rho_critic_1 = torch.stack([input_rho_critic_1, input_rho_critic_with_act_1])
        #     input_rho_critic_2 = torch.stack([input_rho_critic_2, input_rho_critic_with_act_2])
        #
        # with torch.no_grad():
        #     self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.rho_target_critic(input_rho_critic_1, input_rho_critic_2)

        # if actions is not None:
        #     q1_pi_tensor, q2_pi_tensor = self.rho_critic(input_rho_critic_1, input_rho_critic_2)
        #     self.q1_pi_tensor, self.q2_pi_tensor = q1_pi_tensor[0], q2_pi_tensor[0]
        #     return q1_pi_tensor[1], q2_pi_tensor[1]
        #
        # self.q1_pi_tensor, self.q2_pi_tensor = self.rho_critic(input_rho_critic_1, input_rho_critic_2)

        # input_critic = torch.cat([input_actor, repeat_pol_actions], dim=-1)
        # if actions is not None:
        #     repeat_actions = actions.repeat(self.num_blocks, 1, 1)
        #     input_critic_with_act = torch.cat([input_actor, repeat_actions], dim=-1)
        #     input_critic = torch.cat([input_critic, input_critic_with_act], dim=0)
        #
        # with torch.no_grad():
        #     output_phi_target_critic_1, output_phi_target_critic_2 = self.single_phi_target_critic(input_critic[:self.num_blocks])
        #     output_phi_target_critic_1 = output_phi_target_critic_1.sum(dim=0)
        #     output_phi_target_critic_2 = output_phi_target_critic_2.sum(dim=0)
        #     self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.rho_target_critic(output_phi_target_critic_1, output_phi_target_critic_2)
        #
        # output_phi_critic_1, output_phi_critic_2 = self.single_phi_critic(input_critic)
        # if actions is not None:
        #     output_phi_critic_1 = torch.stack([output_phi_critic_1[:self.num_blocks].sum(dim=0),
        #                                        output_phi_critic_1[self.num_blocks:].sum(dim=0)])
        #     output_phi_critic_2 = torch.stack([output_phi_critic_2[:self.num_blocks].sum(dim=0),
        #                                        output_phi_critic_2[self.num_blocks:].sum(dim=0)])
        #     q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
        #     self.q1_pi_tensor, self.q2_pi_tensor = q1_pi_tensor[0], q2_pi_tensor[0]
        #     return q1_pi_tensor[1], q2_pi_tensor[1]
        # else:
        #     output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
        #     output_phi_critic_2 = output_phi_critic_2.sum(dim=0)
        #     self.q1_pi_tensor, self.q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
