import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import permutations
import numpy as np
from rl_modules.networks import GnnMessagePassing, PhiCriticDeepSet, PhiActorDeepSet, RhoActorDeepSet, RhoCriticDeepSet

epsilon = 1e-6


class GnnCritic(nn.Module):
    def __init__(self, nb_objects, aggregation, readout, dim_body, dim_object, dim_mp_input, dim_mp_output, dim_phi_critic_input,
                 dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output):
        super(GnnCritic, self).__init__()

        # self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]
        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object

        self.aggregation = aggregation
        self.readout = readout

        self.mp_critic = GnnMessagePassing(dim_mp_input, dim_mp_output)

        self.isolated_nodes_critic = GnnMessagePassing(15, dim_mp_output)

        self.phi_critic = PhiCriticDeepSet(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_critic = RhoCriticDeepSet(dim_rho_critic_input, dim_rho_critic_output)

        self.edge_ids = [np.array([0, 2]), np.array([1, 4]), np.array([3, 5])]

    def forward(self, obs, act, edge_features, edges_to, isolated_nodes, isolated_nodes_features):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        obs_body = obs[:, :self.dim_body]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        if edge_features is not None:
            if edges_to.shape[0] == 256:
                unrep_unique_obj_ids = torch.tensor([list(set(edges_to[i, :])) for i in range(256)])
                unique_obj_inds = torch.tensor([list(set(edges_to[i, :])) for i in range(256)]).repeat(15, 1, 1).permute(2, 1, 0)
                obs_objects_tensor = torch.stack(obs_objects)
                connected_objects_tensor = torch.gather(obs_objects_tensor, 0, unique_obj_inds)
                incoming = torch.stack([torch.max(edge_features[i, np.where(edges_to[i, :] == int(j))[0], :], dim=0).values
                      for i in range(256) for j in unrep_unique_obj_ids[i, :]])

                incoming = incoming.reshape(connected_objects_tensor.shape[0], connected_objects_tensor.shape[1], incoming.shape[-1])

                rep_body = obs_body.repeat(incoming.shape[0], 1, 1)
                rep_act = act.repeat(incoming.shape[0], 1, 1)
                inp_connected = torch.cat([rep_body, rep_act, connected_objects_tensor, incoming], dim=-1)
            else:
                unique_obj_inds = set(edges_to)
                inp_connected = torch.stack([torch.cat([obs_body, obs_objects[i], torch.max(edge_features[:, np.where(edges_to == i)[0], :],
                                                                                            dim=1).values], dim=1)
                                             for i in unique_obj_inds])

            if isolated_nodes_features is not None:
                inp_isolated = torch.stack([torch.cat([obs_body, act, isolated_nodes[:, i, :], isolated_nodes_features[:, i, :]], dim=1)
                                            for i in range(isolated_nodes.shape[1])])

                inp = torch.cat([inp_connected, inp_isolated], dim=0)
            else:
                inp = inp_connected

        else:
            inp = torch.stack([torch.cat([obs_body, act, isolated_nodes[:, i, :], isolated_nodes_features[:, i, :]], dim=1)
                               for i in range(isolated_nodes.shape[1])])

        output_phi_critic_1, output_phi_critic_2 = self.phi_critic(inp)
        if self.readout == 'sum':
            output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
            output_phi_critic_2 = output_phi_critic_2.sum(dim=0)
        elif self.readout == 'mean':
            output_phi_critic_1 = output_phi_critic_1.mean(dim=0)
            output_phi_critic_2 = output_phi_critic_2.mean(dim=0)
        elif self.readout == 'max':
            output_phi_critic_1 = output_phi_critic_1.max(dim=0).values
            output_phi_critic_2 = output_phi_critic_2.max(dim=0).values
        q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
        return q1_pi_tensor, q2_pi_tensor

    # def message_passing(self, obs, ag, g):
    #     batch_size = obs.shape[0]
    #     assert batch_size == len(ag)
    #
    #     obs_body = obs[:, :self.dim_body]
    #     # obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
    #     #                                       obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
    #     #                            for i in range(self.nb_objects)]
    #     obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
    #                    for i in range(self.nb_objects)]
    #
    #     obj_ids = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]]
    #     goal_ids = [[0, 3], [0, 4], [1, 5], [1, 6], [2, 7], [2, 8]]
    #
    #     inp_mp = torch.stack([torch.cat([ag[:, goal_ids[i]], g[:, goal_ids[i]], obs_objects[obj_ids[i][0]],
    #                                      obs_objects[obj_ids[i][1]]], dim=-1) for i in range(6)])
    #
    #     # inp_mp = torch.stack([torch.cat([g, ag, obj[0], obj[1]], dim=-1) for obj in permutations(obs_objects, 2)])
    #
    #     output_mp = self.mp_critic(inp_mp)
    #
    #     return output_mp


class GnnActor(nn.Module):
    def __init__(self, nb_objects, aggregation, readout, dim_body, dim_object, dim_phi_actor_input, dim_phi_actor_output, dim_rho_actor_input,
                 dim_rho_actor_output):
        super(GnnActor, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object

        self.aggregation = aggregation
        self.readout = readout

        self.phi_actor = PhiActorDeepSet(dim_phi_actor_input, 256, dim_phi_actor_output)
        self.rho_actor = RhoActorDeepSet(dim_rho_actor_input, dim_rho_actor_output)

        self.edge_ids = [np.array([0, 2]), np.array([1, 4]), np.array([3, 5])]

        # self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]

    def forward(self, obs, edge_features, edges_to, isolated_nodes, isolated_nodes_features):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        obs_body = obs[:, :self.dim_body]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        if edge_features is not None:
            if edges_to.shape[0] == 256:
                unrep_unique_obj_ids = torch.tensor([list(set(edges_to[i, :])) for i in range(256)])
                unique_obj_inds = torch.tensor([list(set(edges_to[i, :])) for i in range(256)]).repeat(15, 1, 1).permute(2, 1, 0)
                obs_objects_tensor = torch.stack(obs_objects)
                connected_objects_tensor = torch.gather(obs_objects_tensor, 0, unique_obj_inds)
                incoming = torch.stack([torch.max(edge_features[i, np.where(edges_to[i, :] == int(j))[0], :], dim=0).values
                      for i in range(256) for j in unrep_unique_obj_ids[i, :]])

                incoming = incoming.reshape(connected_objects_tensor.shape[0], connected_objects_tensor.shape[1], incoming.shape[-1])

                rep_body = obs_body.repeat(incoming.shape[0], 1, 1)
                inp_connected = torch.cat([rep_body, connected_objects_tensor, incoming], dim=-1)
            else:
                unique_obj_inds = set(edges_to)
                inp_connected = torch.stack([torch.cat([obs_body, obs_objects[i], torch.max(edge_features[:, np.where(edges_to == i)[0], :],
                                                                                            dim=1).values], dim=1)
                                             for i in unique_obj_inds])

            if isolated_nodes_features is not None:
                inp_isolated = torch.stack([torch.cat([obs_body, isolated_nodes[:, i, :], isolated_nodes_features[:, i, :]], dim=1)
                                            for i in range(isolated_nodes.shape[1])])

                inp = torch.cat([inp_connected, inp_isolated], dim=0)
            else:
                inp = inp_connected

        else:
            inp = torch.stack([torch.cat([obs_body, isolated_nodes[:, i, :], isolated_nodes_features[:, i, :]], dim=1)
                               for i in range(isolated_nodes.shape[1])])

        output_phi_actor = self.phi_actor(inp)
        if self.readout == 'sum':
            output_phi_actor = output_phi_actor.sum(dim=0)
        elif self.readout == 'mean':
            output_phi_actor = output_phi_actor.mean(dim=0)
        elif self.readout == 'max':
            output_phi_actor = output_phi_actor.max(dim=0).values
        else:
            raise NotImplementedError

        mean, logstd = self.rho_actor(output_phi_actor)
        return mean, logstd

    def sample(self, obs, edge_features, edges_to, isolated_nodes, isolated_nodes_features):
        mean, log_std = self.forward(obs, edge_features, edges_to, isolated_nodes, isolated_nodes_features)
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
        self.dim_body = env_params['body_dim']
        self.dim_object = env_params['obj_dim']
        self.dim_goal = env_params['goal_dim']
        self.dim_act = env_params['action_dim']
        self.nb_objects = env_params['n_blocks']

        self.num_predicates = env_params['n_predicates']

        self.aggregation = args.aggregation_fct
        self.readout = args.readout_fct

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        # dim_input_objects = 2 * (self.nb_objects + self.dim_object)
        dim_mp_input = 2 * (self.dim_object + self.num_predicates)
        dim_mp_output = 3 * dim_mp_input

        dim_phi_actor_input = self.dim_body + self.dim_object + dim_mp_output
        dim_phi_actor_output = 3 * dim_phi_actor_input
        dim_rho_actor_input = dim_phi_actor_output
        dim_rho_actor_output = self.dim_act

        dim_phi_critic_input = self.dim_body + self.dim_object + dim_mp_output + self.dim_act
        dim_phi_critic_output = 3 * dim_phi_critic_input
        dim_rho_critic_input = dim_phi_critic_output
        dim_rho_critic_output = 1

        self.critic = GnnCritic(self.nb_objects, self.aggregation, self.readout, self.dim_body, self.dim_object, dim_mp_input, dim_mp_output,
                                dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output)
        self.critic_target = GnnCritic(self.nb_objects, self.aggregation, self.readout, self.dim_body, self.dim_object, dim_mp_input, dim_mp_output,
                                       dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output)
        self.actor = GnnActor(self.nb_objects, self.aggregation, self.readout, self.dim_body, self.dim_object, dim_phi_actor_input,
                              dim_phi_actor_output, dim_rho_actor_input, dim_rho_actor_output)

    def policy_forward_pass(self, obs, graph, edges_to, isolated_nodes, no_noise=False):
        # edge_features = self.critic.message_passing(obs, ag, g)
        if graph.shape[1] != 0:
            edge_features = self.critic.mp_critic(graph)
        else:
            edge_features = None
        if isolated_nodes.shape[1] != 0:
            isolated_nodes_features = self.critic.isolated_nodes_critic(isolated_nodes)
        else:
            isolated_nodes_features = None
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, edge_features, edges_to, isolated_nodes, isolated_nodes_features)
        else:
            _, self.log_prob, self.pi_tensor = self.actor.sample(obs, edge_features, edges_to, isolated_nodes, isolated_nodes_features)

    def forward_pass(self, obs, graph, edges_to, isolated_nodes, actions=None):
        # edge_features = self.critic.message_passing(obs, ag, g)

        if graph.shape[1] != 0:
            edge_features = self.critic.mp_critic(graph)
        else:
            edge_features = None
        if isolated_nodes.shape[1] != 0:
            isolated_nodes_features = self.critic.isolated_nodes_critic(isolated_nodes)
        else:
            isolated_nodes_features = None

        self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, edge_features, edges_to, isolated_nodes, isolated_nodes_features)

        if actions is not None:
            self.q1_pi_tensor, self.q2_pi_tensor = self.critic.forward(obs, self.pi_tensor, edge_features, edges_to, isolated_nodes,
                                                                       isolated_nodes_features)
            return self.critic.forward(obs, actions, edge_features, edges_to, isolated_nodes, isolated_nodes_features)
        else:
            with torch.no_grad():
                self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.critic_target.forward(obs, self.pi_tensor, edge_features,
                                                                                                edges_to, isolated_nodes, isolated_nodes_features)
            self.q1_pi_tensor, self.q2_pi_tensor = None, None
