import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import permutations
import numpy as np
from rl_modules.networks import GnnMessagePassing, PhiCriticDeepSet, PhiActorDeepSet, RhoActorDeepSet, RhoCriticDeepSet

epsilon = 1e-6


class GnnCritic(nn.Module):
    def __init__(self, nb_objects, nb_permutations, dim_body, dim_object, dim_mp_input, dim_mp_output, dim_phi_critic_input,
                 dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output):
        super(GnnCritic, self).__init__()

        self.nb_permutations = nb_permutations
        # self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]
        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object

        self.mp_critic = GnnMessagePassing(dim_mp_input, dim_mp_output)
        self.phi_critic = PhiActorDeepSet(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_critic = RhoCriticDeepSet(dim_rho_critic_input, dim_rho_critic_output)

        self.mlp = PhiActorDeepSet(14, 256, 42)

        self.edge_ids = [np.array([0, 2]), np.array([1, 4]), np.array([3, 5])]

    def forward(self, obs, act, graph_features):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        obs_body = obs[:, :self.dim_body]
        # # obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
        # #                           obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
        # #                for i in range(self.nb_objects)]
        # obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
        #                for i in range(self.nb_objects)]
        #
        # inp = torch.stack([torch.cat([obj, torch.max(edge_features[self.edge_ids[i], :, :], dim=0).values], dim=1)
        #                    for i, obj in enumerate(obs_objects)])
        #
        # output_phi_critic_1, output_phi_critic_2 = self.phi_critic(inp)
        # output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
        # output_phi_critic_2 = output_phi_critic_2.sum(dim=0)

        inp_mlp = torch.cat([obs_body, act], dim=1)

        global_features = self.mlp(inp_mlp)

        inp_rho_critic_1 = torch.cat([graph_features, global_features], dim=1)
        inp_rho_critic_2 = torch.cat([graph_features, global_features], dim=1)

        q1_pi_tensor, q2_pi_tensor = self.rho_critic(inp_rho_critic_1, inp_rho_critic_2)
        return q1_pi_tensor, q2_pi_tensor

    def message_passing(self, obs, ag, g):
        batch_size = obs.shape[0]
        assert batch_size == len(ag)

        obs_body = obs[:, :self.dim_body]
        # obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
        #                                       obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
        #                            for i in range(self.nb_objects)]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        obj_ids = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]]
        goal_ids = [[0, 3], [0, 4], [1, 5], [1, 6], [2, 7], [2, 8]]

        inp_mp = torch.stack([torch.cat([ag[:, goal_ids[i]], g[:, goal_ids[i]], obs_objects[obj_ids[i][0]][:, :3],
                                         obs_objects[obj_ids[i][1]][:, :3]], dim=-1) for i in range(6)])

        output_mp = self.mp_critic(inp_mp)

        return output_mp

    def graph_forward(self, obs, edge_features):
        obs_body = obs[:, :self.dim_body]
        # obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
        #                           obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
        #                for i in range(self.nb_objects)]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        inp = torch.stack([torch.cat([obj, torch.max(edge_features[self.edge_ids[i], :, :], dim=0).values], dim=1)
                           for i, obj in enumerate(obs_objects)])

        output_phi_critic = self.phi_critic(inp)

        return output_phi_critic.sum(dim=0)


class GnnActor(nn.Module):
    def __init__(self, nb_objects, dim_body, dim_object, dim_phi_actor_input, dim_phi_actor_output, dim_rho_actor_input, dim_rho_actor_output):
        super(GnnActor, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object

        # self.phi_actor = PhiActorDeepSet(dim_phi_actor_input, 256, dim_phi_actor_output)
        self.rho_actor = RhoActorDeepSet(dim_rho_actor_input, dim_rho_actor_output)

        self.mlp = PhiActorDeepSet(10, 256, 30)

        self.edge_ids = [np.array([0, 2]), np.array([1, 4]), np.array([3, 5])]

        # self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]

    def forward(self, obs, graph_features):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        obs_body = obs[:, :self.dim_body]
        # obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
        #                           obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
        #                for i in range(self.nb_objects)]
        # obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
        #                for i in range(self.nb_objects)]
        #
        # inp = torch.stack([torch.cat([obj, torch.max(edge_features[self.edge_ids[i], :, :], dim=0).values], dim=1)
        #                               for i, obj in enumerate(obs_objects)])
        #
        # output_phi_actor = self.phi_actor(inp)
        # output_phi_actor = output_phi_actor.sum(dim=0)

        global_feature = self.mlp(obs_body)

        inp_rho_actor = torch.cat([graph_features, global_feature], dim=1)

        mean, logstd = self.rho_actor(inp_rho_actor)
        return mean, logstd

    def sample(self, obs, edge_features):
        mean, log_std = self.forward(obs, edge_features)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class GnnSemantic:
    def __init__(self, env_params, args):
        self.dim_body = 10
        self.dim_object = 15
        self.dim_goal = env_params['goal']
        self.dim_act = env_params['action']
        self.nb_objects = 3
        self.n_permutations = 3

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        # dim_input_objects = 2 * (self.nb_objects + self.dim_object)
        dim_mp_input = 6 + 4
        dim_mp_output = 3 * dim_mp_input

        dim_phi_actor_input = self.dim_object + dim_mp_output
        dim_phi_actor_output = 3 * dim_phi_actor_input
        dim_rho_actor_input = dim_phi_actor_output + 30
        dim_rho_actor_output = self.dim_act

        dim_phi_critic_input = self.dim_object + dim_mp_output
        dim_phi_critic_output = 3 * dim_phi_critic_input
        dim_rho_critic_input = dim_phi_critic_output + 42
        dim_rho_critic_output = 1

        self.critic = GnnCritic(self.nb_objects, self.n_permutations, self.dim_body, self.dim_object, dim_mp_input, dim_mp_output,
                                dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output)
        self.critic_target = GnnCritic(self.nb_objects, self.n_permutations, self.dim_body, self.dim_object, dim_mp_input, dim_mp_output,
                                       dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output)
        self.actor = GnnActor(self.nb_objects, self.dim_body, self.dim_object, dim_phi_actor_input, dim_phi_actor_output, dim_rho_actor_input,
                              dim_rho_actor_output)

    def policy_forward_pass(self, obs, ag, g, no_noise=False):
        edge_features = self.critic.message_passing(obs, ag, g)
        graph_features = self.critic.graph_forward(obs, edge_features)
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, graph_features)
        else:
            _, self.log_prob, self.pi_tensor = self.actor.sample(obs, graph_features)

    def forward_pass(self, obs, ag, g, actions=None):
        edge_features = self.critic.message_passing(obs, ag, g)
        graph_features = self.critic.graph_forward(obs, edge_features)

        self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, graph_features)

        if actions is not None:
            self.q1_pi_tensor, self.q2_pi_tensor = self.critic.forward(obs, self.pi_tensor, graph_features)
            return self.critic.forward(obs, actions, graph_features)
        else:
            with torch.no_grad():
                self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.critic_target.forward(obs, self.pi_tensor, graph_features)
            self.q1_pi_tensor, self.q2_pi_tensor = None, None