import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import permutations
import numpy as np
# from rl_modules.networks import PhiActorDeepSet, PhiCriticDeepSet, RhoActorDeepSet, RhoCriticDeepSet
from rl_modules.networks import GnnAttention, GnnMessagePassing, PhiCriticDeepSet, PhiActorDeepSet, RhoActorDeepSet, RhoCriticDeepSet

epsilon = 1e-6


class GnnCritic(nn.Module):
    def __init__(self, nb_objects, nb_permutations, dim_body, dim_object):
        super(GnnCritic, self).__init__()

        self.nb_permutations = nb_permutations
        self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]
        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object

        dim_attention_input = 2 * 9
        dim_attention_output = nb_objects

        dim_input_objects = 2 * (self.nb_objects + self.dim_object)
        dim_mp_input = dim_input_objects + 2 * 3
        dim_mp_output = 3 * dim_mp_input

        dim_phi_critic_input = self.dim_body + (self.nb_objects + self.dim_object) + dim_mp_output + 4
        dim_phi_critic_output = 3 * dim_phi_critic_input
        dim_rho_critic_input = dim_phi_critic_output
        dim_rho_critic_output = 1

        self.gnn_attention = GnnAttention(dim_attention_input, 64, dim_attention_output)
        self.mp_critic = GnnMessagePassing(dim_mp_input, dim_mp_output)
        self.phi_critic = PhiCriticDeepSet(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_critic = RhoCriticDeepSet(dim_rho_critic_input, dim_rho_critic_output)

        self.goal_ids = [torch.tensor([0., 3., 4., 1., 5., 6.]), torch.tensor([0., 3., 4., 2., 7., 8.]), torch.tensor([1., 5., 6., 2., 7., 8.])]

    def forward(self, inp, act):
        rep_act = act.repeat(3, 1, 1)
        inp = torch.cat([rep_act, inp], dim=-1)
        output_phi_critic_1, output_phi_critic_2 = self.phi_critic(inp)
        output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
        output_phi_critic_2 = output_phi_critic_2.sum(dim=0)
        q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
        return q1_pi_tensor, q2_pi_tensor

    def gumbel_attention(self, g, ag):
        inp_attention = torch.cat([g, ag], dim=1)
        return self.gnn_attention(inp_attention)

    def message_passing(self, obs, ag, g, attended_ids):
        batch_size = obs.shape[0]
        assert batch_size == len(ag)

        obs_body = obs[:, :self.dim_body]
        obs_objects = torch.stack([torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
                                              obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
                                   for i in range(self.nb_objects)]).permute(1, 0, 2)
        # Attention
        pivot_nodes = torch.matmul(attended_ids, obs_objects)
        remaining_nodes = obs_objects[:, torch.nonzero(attended_ids - 1)[:, -1], :]
        attended_edges = torch.matmul(attended_ids, torch.stack([self.goal_ids[i] for i in range(self.nb_objects)]))

        inp_mp_pivot = torch.stack([torch.cat([torch.gather(g, -1, attended_edges[:, :3].long()),
                                               torch.gather(ag, -1, attended_edges[:, :3].long()), pivot_nodes[:, 0, :],
                                               remaining_nodes[:, i, :]], dim=-1) for i in range(remaining_nodes.shape[1])])
        inp_mp_remain = torch.stack([torch.cat([torch.gather(g, -1, attended_edges[:, :3].long()),
                                                torch.gather(ag, -1, attended_edges[:, :3].long()), remaining_nodes[:, i, :],
                                                pivot_nodes[:, 0, :]], dim=-1) for i in range(remaining_nodes.shape[1])])

        inp_mp = torch.cat([inp_mp_pivot, inp_mp_remain], dim=0)

        output_mp = self.mp_critic(inp_mp)

        inp = torch.stack([torch.cat([obs_body, pivot_nodes[:, 0, :], torch.max(output_mp[:2, :, :], dim=0).values], dim=1),
                           torch.cat([obs_body, remaining_nodes[:, 0, :], output_mp[-2, :, :]], dim=1),
                           torch.cat([obs_body, remaining_nodes[:, 1, :], output_mp[-1, :, :]], dim=1)])

        return inp


class GnnActor(nn.Module):
    def __init__(self, nb_objects, dim_body, dim_object):
        super(GnnActor, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object

        dim_input_objects = 2 * (self.nb_objects + self.dim_object)
        dim_mp_input = dim_input_objects + 2 * 3
        dim_mp_output = 3 * dim_mp_input

        dim_phi_actor_input = self.dim_body + (self.nb_objects + self.dim_object) + dim_mp_output
        dim_phi_actor_output = 3 * dim_phi_actor_input
        dim_rho_actor_input = dim_phi_actor_output
        dim_rho_actor_output = 4

        # self.mp_actor = GnnMessagePassing(dim_mp_input, dim_mp_output)
        self.phi_actor = PhiActorDeepSet(dim_phi_actor_input, 256, dim_phi_actor_output)
        self.rho_actor = RhoActorDeepSet(dim_rho_actor_input, dim_rho_actor_output)

        self.goal_ids = [torch.tensor([0., 3., 4., 1., 5., 6.]), torch.tensor([0., 3., 4., 2., 7., 8.]), torch.tensor([1., 5., 6., 2., 7., 8.])]

        self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]

    def forward(self, inp):
        output_phi_actor = self.phi_actor(inp)
        output_phi_actor = output_phi_actor.sum(dim=0)
        mean, logstd = self.rho_actor(output_phi_actor)
        return mean, logstd

    def sample(self, inp):
        mean, log_std = self.forward(inp)
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
        self.observation = None
        self.ag = None
        self.g = None
        self.dim_body = 10
        self.dim_object = 15
        self.dim_goal = env_params['goal']
        self.dim_act = env_params['action']
        self.nb_objects = 3
        self.n_permutations = len([x for x in permutations(range(self.nb_objects), 2)])

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        # dim_input_objects = 2 * (self.nb_objects + self.dim_object)
        # dim_phi_actor_input = 2 * self.dim_goal + self.dim_body + dim_input_objects
        # dim_phi_actor_output = 3 * dim_phi_actor_input
        # dim_rho_actor_input = dim_phi_actor_output
        # dim_rho_actor_output = self.dim_act
        #
        # dim_phi_critic_input = dim_phi_actor_input + self.dim_act
        # dim_phi_critic_output = 3 * dim_phi_critic_input
        # dim_rho_critic_input = dim_phi_critic_output
        # dim_rho_critic_output = 1

        self.critic = GnnCritic(self.nb_objects, self.n_permutations, self.dim_body, self.dim_object)
        self.critic_target = GnnCritic(self.nb_objects, self.n_permutations, self.dim_body, self.dim_object)
        self.actor = GnnActor(self.nb_objects, self.dim_body, self.dim_object)

    def policy_forward_pass(self, obs, ag, g, anchor_g, no_noise=False):
        attended_ids = self.critic.gumbel_attention(g, ag)
        inp = self.critic.message_passing(obs, ag, g, attended_ids)
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.actor.sample(inp)
        else:
            _, self.log_prob, self.pi_tensor = self.actor.sample(inp)

    def forward_pass(self, obs, ag, g, anchor_g, actions=None):
        attended_ids = self.critic.gumbel_attention(g, ag)
        inp = self.critic.message_passing(obs, ag, g, attended_ids)

        self.pi_tensor, self.log_prob, _ = self.actor.sample(inp)

        if actions is not None:
            self.q1_pi_tensor, self.q2_pi_tensor = self.critic.forward(inp, self.pi_tensor)
            return self.critic.forward(inp, actions)
        else:
            with torch.no_grad():
                self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.critic_target.forward(inp, self.pi_tensor)
            self.q1_pi_tensor, self.q2_pi_tensor = None, None
