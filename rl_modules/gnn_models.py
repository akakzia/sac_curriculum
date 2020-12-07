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
        self.phi_critic = PhiCriticDeepSet(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_critic = RhoCriticDeepSet(dim_rho_critic_input, dim_rho_critic_output)

        # self.edge_ids = [np.array([0, 2]), np.array([1, 4]), np.array([3, 5])]
        self.edge_ids = [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11]), np.array([12, 13, 14, 15]),
                         np.array([16, 17, 18, 19])]

    def forward(self, obs, act, edge_features):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        obs_body = obs[:, :self.dim_body]
        # obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
        #                           obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
        #                for i in range(self.nb_objects)]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        inp = torch.stack([torch.cat([act, obs_body, obj, torch.max(edge_features[self.edge_ids[i], :, :], dim=0).values], dim=1)
                           for i, obj in enumerate(obs_objects)])

        output_phi_critic_1, output_phi_critic_2 = self.phi_critic(inp)
        output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
        output_phi_critic_2 = output_phi_critic_2.sum(dim=0)
        q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
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

        # obj_ids = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]]
        # goal_ids = [[0, 3], [0, 4], [1, 5], [1, 6], [2, 7], [2, 8]]
        obj_ids = list(permutations(np.arange(self.nb_objects), 2))
        goal_ids = [[0, 10], [1, 11], [2, 12], [3, 13], [0, 14], [4, 15], [5, 16], [6, 17], [1, 18], [4, 19], [7, 20], [8, 21], [2, 22], [5, 23],
                    [7, 24], [9, 25], [3, 26], [6, 27], [8, 28], [9, 29]]

        inp_mp = torch.stack([torch.cat([ag[:, goal_ids[i]], g[:, goal_ids[i]], obs_objects[obj_ids[i][0]][:, :3],
                                         obs_objects[obj_ids[i][1]][:, :3]], dim=-1) for i in range(len(obj_ids))])

        # inp_mp = torch.stack([torch.cat([g, ag, obj[0], obj[1]], dim=-1) for obj in permutations(obs_objects, 2)])

        output_mp = self.mp_critic(inp_mp)

        return output_mp


class GnnActor(nn.Module):
    def __init__(self, nb_objects, dim_body, dim_object, dim_phi_actor_input, dim_phi_actor_output, dim_rho_actor_input, dim_rho_actor_output):
        super(GnnActor, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object

        self.phi_actor = PhiActorDeepSet(dim_phi_actor_input, 256, dim_phi_actor_output)
        self.rho_actor = RhoActorDeepSet(dim_rho_actor_input, dim_rho_actor_output)

        # self.edge_ids = [np.array([0, 2]), np.array([1, 4]), np.array([3, 5])]
        self.edge_ids = [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11]), np.array([12, 13, 14, 15]),
                         np.array([16, 17, 18, 19])]

        # self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]

    def forward(self, obs, edge_features):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        obs_body = obs[:, :self.dim_body]
        # obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
        #                           obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
        #                for i in range(self.nb_objects)]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        inp = torch.stack([torch.cat([obs_body, obj, torch.max(edge_features[self.edge_ids[i], :, :], dim=0).values], dim=1)
                                   for i, obj in enumerate(obs_objects)])

        output_phi_actor = self.phi_actor(inp)
        output_phi_actor = output_phi_actor.sum(dim=0)
        mean, logstd = self.rho_actor(output_phi_actor)
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
        self.nb_objects = env_params['num_objects']
        self.n_permutations = self.nb_objects

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        # dim_input_objects = 2 * (self.nb_objects + self.dim_object)
        dim_mp_input = 6 + 4
        dim_mp_output = 3 * dim_mp_input

        dim_phi_actor_input = self.dim_body + self.dim_object + dim_mp_output
        dim_phi_actor_output = 3 * dim_phi_actor_input
        dim_rho_actor_input = dim_phi_actor_output
        dim_rho_actor_output = self.dim_act

        dim_phi_critic_input = self.dim_body + self.dim_object + dim_mp_output + self.dim_act
        dim_phi_critic_output = 3 * dim_phi_critic_input
        dim_rho_critic_input = dim_phi_critic_output
        dim_rho_critic_output = 1

        self.critic = GnnCritic(self.nb_objects, self.n_permutations, self.dim_body, self.dim_object, dim_mp_input, dim_mp_output,
                                dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output)
        self.critic_target = GnnCritic(self.nb_objects, self.n_permutations, self.dim_body, self.dim_object, dim_mp_input, dim_mp_output,
                                       dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output)
        self.actor = GnnActor(self.nb_objects, self.dim_body, self.dim_object, dim_phi_actor_input, dim_phi_actor_output, dim_rho_actor_input,
                              dim_rho_actor_output)

    def policy_forward_pass(self, obs, ag, g, no_noise=False):
        edge_features = self.critic.message_passing(obs, ag, g)
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, edge_features)
        else:
            _, self.log_prob, self.pi_tensor = self.actor.sample(obs, edge_features)

    def forward_pass(self, obs, ag, g, actions=None):
        edge_features = self.critic.message_passing(obs, ag, g)

        self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, edge_features)

        if actions is not None:
            self.q1_pi_tensor, self.q2_pi_tensor = self.critic.forward(obs, self.pi_tensor, edge_features)
            return self.critic.forward(obs, actions, edge_features)
        else:
            with torch.no_grad():
                self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.critic_target.forward(obs, self.pi_tensor, edge_features)
            self.q1_pi_tensor, self.q2_pi_tensor = None, None