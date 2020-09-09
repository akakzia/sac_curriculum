import torch
import numpy as np
from mpi_utils.mpi_utils import sync_networks
from rl_modules.replay_buffer import MultiBuffer
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from rl_modules.td3_deepset_models import DeepSetTD3
from updates import update_deepsets_td3

"""
TD3 with HER and deep sets
"""


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class TD3Agent:
    def __init__(self, args, compute_rew, goal_sampler):

        self.args = args
        self.env_params = args.env_params
        self.total_it = 0
        self.policy_freq = 2

        self.goal_sampler = goal_sampler

        # create the network
        self.architecture = self.args.architecture
        if self.architecture == 'deepsets':
            self.model = DeepSetTD3(self.env_params, args)
            # sync the networks across the CPUs
            sync_networks(self.model.rho_actor)
            sync_networks(self.model.rho_critic)
            sync_networks(self.model.single_phi_actor)
            sync_networks(self.model.single_phi_critic)

            # Define target critic
            hard_update(self.model.single_phi_target_critic, self.model.single_phi_critic)
            hard_update(self.model.rho_target_critic, self.model.rho_critic)

            # Define target actor
            hard_update(self.model.single_phi_target_actor, self.model.single_phi_actor)
            hard_update(self.model.rho_target_actor, self.model.rho_actor)

            # sync targets
            sync_networks(self.model.single_phi_target_critic)
            sync_networks(self.model.rho_target_critic)
            # create the optimizer
            self.policy_optim = torch.optim.Adam(list(self.model.single_phi_actor.parameters()) +
                                                 list(self.model.rho_actor.parameters()),
                                                 lr=self.args.lr_actor)
            self.critic_optim = torch.optim.Adam(list(self.model.single_phi_critic.parameters()) +
                                                 list(self.model.rho_critic.parameters()),
                                                 lr=self.args.lr_critic)

        else:
            raise NotImplementedError

        # create the normalizer
        self.o_norm = normalizer(size=self.env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=self.env_params['goal'], default_clip_range=self.args.clip_range)

        # her sampler
        self.continuous_goals = False
        self.language = False

        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, compute_rew)

        # create the replay buffer
        self.buffer = MultiBuffer(env_params=self.env_params,
                                  buffer_size=self.args.buffer_size,
                                  sample_func=self.her_module.sample_her_transitions,
                                  multi_head=self.args.multihead_buffer,
                                  goal_sampler=self.goal_sampler
                                  )

    def act(self, obs, ag, g, no_noise):
        with torch.no_grad():
            # normalize policy inputs
            obs_norm = self.o_norm.normalize(obs)
            ag_norm = torch.tensor(self.g_norm.normalize(ag), dtype=torch.float32).unsqueeze(0)

            if self.language:
                g_norm = g
            else:
                g_norm = torch.tensor(self.g_norm.normalize(g), dtype=torch.float32).unsqueeze(0)
            if self.architecture == 'deepsets':
                obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
                self.model.policy_forward_pass(obs_tensor, ag_norm, g_norm, no_noise=no_noise)
                action = self.model.pi_tensor.numpy()[0]

            else:
                input_tensor = self._preproc_inputs(obs, g)
                action = self._select_actions(input_tensor, no_noise=no_noise)

        return action.copy()

    def store(self, episodes):
        self.buffer.store_episode(episode_batch=episodes)

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    def train(self):
        # train the network
        self.total_it += 1
        self._update_network()

        # soft update
        if self.total_it % self.policy_freq == 0:
            if self.architecture == 'deepsets':
                self._soft_update_target_network(self.model.single_phi_target_critic, self.model.single_phi_critic)
                self._soft_update_target_network(self.model.rho_target_critic, self.model.rho_critic)

                self._soft_update_target_network(self.model.single_phi_target_actor, self.model.single_phi_actor)
                self._soft_update_target_network(self.model.rho_target_actor, self.model.rho_actor)
            else:
                raise NotImplementedError

    def _select_actions(self, state, no_noise=False):
        if not no_noise:
            action, _, _ = self.actor_network.sample(state)
        else:
            _, _, action = self.actor_network.sample(state)
        return action.detach().cpu().numpy()[0]

    # update the normalizer
    def _update_normalizer(self, episode):
        mb_obs = episode['obs']
        mb_ag = episode['ag']
        mb_g = episode['g']
        mb_actions = episode['act']
        mb_obs_next = mb_obs[1:, :]
        mb_ag_next = mb_ag[1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[0]
        # create the new buffer to store them
        buffer_temp = {'obs': np.expand_dims(mb_obs, 0),
                       'ag': np.expand_dims(mb_ag, 0),
                       'g': np.expand_dims(mb_g, 0),
                       'actions': np.expand_dims(mb_actions, 0),
                       'obs_next': np.expand_dims(mb_obs_next, 0),
                       'ag_next': np.expand_dims(mb_ag_next, 0),
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        # recompute the stats
        self.o_norm.recompute_stats()

        if self.args.normalize_goal:
            self.g_norm.update(transitions['g'])
            self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):

        # sample from buffer, this is done with LP is multi-head is true
        transitions = self.buffer.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next, g, ag, ag_next, actions, rewards = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['ag'], \
                                                      transitions['ag_next'], transitions['actions'], transitions['r']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        _, transitions['ag'] = self._preproc_og(o, ag)
        _, transitions['ag_next'] = self._preproc_og(o, ag_next)

        # apply normalization
        obs_norm = self.o_norm.normalize(transitions['obs'])
        if self.language:
            g_norm = transitions['g']
        else:
            g_norm = self.g_norm.normalize(transitions['g'])
        ag_norm = self.g_norm.normalize(transitions['ag'])
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        ag_next_norm = self.g_norm.normalize(transitions['ag_next'])
        # g_next_norm = self.g_norm.normalize(transitions['g_next'])

        if self.architecture == 'deepsets':
            update_deepsets_td3(self.model, self.language, self.policy_optim, self.critic_optim, obs_norm, ag_norm, g_norm, obs_next_norm,
                                ag_next_norm, actions, rewards, self.args, self.total_it, self.policy_freq)
        else:
            raise NotImplementedError

    def save(self, model_path, epoch):
        # Store model
        if self.args.architecture == 'deepsets':
            torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                        self.model.single_phi_actor.state_dict(), self.model.single_phi_critic.state_dict(),
                        self.model.rho_actor.state_dict(), self.model.rho_critic.state_dict()],
                       model_path + '/model_{}.pt'.format(epoch))
        else:
            raise NotImplementedError

    def load(self, model_path, args):

        if args.architecture == 'deepsets':
            o_mean, o_std, g_mean, g_std, phi_a, phi_c, rho_a, rho_c = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model.single_phi_actor.load_state_dict(phi_a)
            self.model.single_phi_critic.load_state_dict(phi_c)
            self.model.rho_actor.load_state_dict(rho_a)
            self.model.rho_critic.load_state_dict(rho_c)
            self.o_norm.mean = o_mean
            self.o_norm.std = o_std
            self.g_norm.mean = g_mean
            self.g_norm.std = g_std
        else:
            raise NotImplementedError
