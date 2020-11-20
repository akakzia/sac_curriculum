import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import permutations, combinations
import numpy as np
from language.utils import OneHotEncoder, analyze_inst, Vocab
from utils import get_instruction, get_instruction2

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

ONE_HOT = False
UNIQUE_ENCODER = True

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


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


class DeepSetSAC:
    def __init__(self, env_params, args):
        # A raw version of DeepSet-based SAC without attention mechanism
        self.observation = None
        self.ag = None
        self.g = None
        self.dim_body = 10
        self.dim_object = 15
        self.dim_goal = env_params['goal']
        self.dim_act = env_params['action']
        self.num_blocks = 3
        self.combinations_trick = args.combinations_trick
        if self.combinations_trick:
            self.n_permutations = len([x for x in combinations(range(self.num_blocks), 2)])
        else:
            self.n_permutations = len([x for x in permutations(range(self.num_blocks), 2)])
        # self.n_permutations = 1



        if args.algo == 'continuous' or args.algo == 'language':
            self.symmetry_trick = False
            self.include_ag = False
            self.continuous_trick = args.continuous_trick
            if self.continuous_trick:
                self.obj_ids = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
            self.combinations_trick = False
        else:
            self.include_ag = True
            self.continuous_trick = False
            self.symmetry_trick = args.symmetry_trick
        if self.symmetry_trick:
            self.first_inds = np.array([0, 1, 2, 3, 5, 7])
            self.second_inds = np.array([0, 1, 2, 4, 6, 8])
            self.dim_goal = 6
        elif self.combinations_trick:
            self.dim_goal = 2

        if args.algo == 'language':
            self.language = True
            self.instructions = get_instruction2()
            self.nb_instructions = len(self.instructions)
            self.embedding_size = self.nb_instructions if ONE_HOT else args.embedding_size


            split_instructions, max_seq_length, word_set = analyze_inst(self.instructions)
            vocab = Vocab(word_set)
            self.one_hot_encoder = OneHotEncoder(vocab, max_seq_length)
            self.one_hot_language = dict(zip(self.instructions, [self.one_hot_encoder.encode(s) for s in split_instructions]))

            if ONE_HOT:
                onehots = []
                for i in range(self.nb_instructions):
                    zeros = np.zeros([self.nb_instructions])
                    zeros[i] = 1
                    onehots.append(zeros)
                self.simple_encoder = dict(zip(self.instructions, onehots))

            elif not UNIQUE_ENCODER:
                self.policy_sentence_encoder = nn.RNN(input_size=len(word_set) + 1,
                                                      hidden_size=self.embedding_size,
                                                      num_layers=1,
                                                      nonlinearity='tanh',
                                                      bias=True,
                                                      batch_first=True)

            self.critic_sentence_encoder = nn.RNN(input_size=len(word_set) + 1,
                                                  hidden_size=self.embedding_size,
                                                  num_layers=1,
                                                  nonlinearity='tanh',
                                                  bias=True,
                                                  batch_first=True)

            self.target_critic_sentence_encoder = nn.RNN(input_size=len(word_set) + 1,
                                                  hidden_size=self.embedding_size,
                                                  num_layers=1,
                                                  nonlinearity='tanh',
                                                  bias=True,
                                                  batch_first=True)

            # OLD language model
            # self.language = True
            # self.instruction_dict, self.g_str_to_inst = get_instruction()
            # sentences = list(self.instruction_dict.values())
            #
            # set_sentences = set(sentences)
            # split_instructions, max_seq_length, word_set = analyze_inst(set_sentences)
            # vocab = Vocab(word_set)
            # self.one_hot_encoder = OneHotEncoder(vocab, max_seq_length)
            # self.one_hot_language = dict(zip(self.g_str_to_inst.keys(), [self.one_hot_encoder.encode(s) for s in split_instructions]))
            #
            # self.policy_sentence_encoder = nn.RNN(input_size=len(word_set) + 1,
            #                                       hidden_size=100,
            #                                       num_layers=1,
            #                                       nonlinearity='tanh',
            #                                       bias=True,
            #                                       batch_first=True)
            #
            # self.critic_sentence_encoder = nn.RNN(input_size=len(word_set) + 1,
            #                                       hidden_size=100,
            #                                       num_layers=1,
            #                                       nonlinearity='tanh',
            #                                       bias=True,
            #                                       batch_first=True)
        else:
            self.language = False

        # double_critic_attention = double_critic_attention
        self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        # Define dimensions
        if self.language:
            dim_input_goals = self.embedding_size
        else:
            if self.include_ag:
                dim_input_goals = 2 * self.dim_goal
            else:
                dim_input_goals = self.dim_goal

        dim_input_objects = 2 * (self.num_blocks + self.dim_object)
        if self.continuous_trick:
            dim_phi_actor_input = 2*6 + self.dim_body + dim_input_objects
        else:
            dim_phi_actor_input = dim_input_goals + self.dim_body + dim_input_objects + 3 # for distances

        # dim_phi_actor_output = 3 * (self.dim_body + (self.num_blocks + self.dim_object))
        dim_phi_actor_output = 3 * dim_phi_actor_input

        dim_rho_actor_input = dim_phi_actor_output
        dim_rho_actor_output = self.dim_act

        if self.continuous_trick:
            dim_phi_critic_input = 6*2 + self.dim_body + dim_input_objects + self.dim_act
        else:
            dim_phi_critic_input = dim_input_goals + self.dim_body + dim_input_objects + self.dim_act + 3 # for distances

        # dim_phi_critic_output = 3 * (self.dim_body + (self.num_blocks + self.dim_object) + self.dim_act)
        dim_phi_critic_output = 3 * dim_phi_critic_input

        dim_rho_critic_input = dim_phi_critic_output
        dim_rho_critic_output = 1

        self.single_phi_actor = SinglePhiActor(dim_phi_actor_input, 256, dim_phi_actor_output)
        self.rho_actor = RhoActor(dim_rho_actor_input, dim_rho_actor_output)

        self.single_phi_critic = SinglePhiCritic(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_critic = RhoCritic(dim_rho_critic_input, dim_rho_critic_output)

        self.single_phi_target_critic = SinglePhiCritic(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_target_critic = RhoCritic(dim_rho_critic_input, dim_rho_critic_output)

    def policy_forward_pass(self, obs, ag, g, anchor_g=None, no_noise=False, language_goal=None):
        self.observation = obs
        self.ag = ag
        self.g = g

        self.anchor_g = anchor_g

        if self.language:
            if ONE_HOT:
                goal_embeddings = torch.tensor(self.simple_encoder[language_goal], dtype=torch.float32).unsqueeze(0)
            else:
                encodings = self.one_hot_language[language_goal]
                # old
                # encodings = np.array(self.one_hot_language[str(g)])
                encodings = torch.tensor(encodings, dtype=torch.float32).unsqueeze(0)
                # goal_embeddings = self.policy_sentence_encoder.forward(encodings)[0][:, -1, :]
                if UNIQUE_ENCODER:
                    goal_embeddings = self.critic_sentence_encoder.forward(encodings)[0][:, -1, :]
                else:
                    goal_embeddings = self.policy_sentence_encoder.forward(encodings)[0][:, -1, :]

        obs_distances = self.observation.narrow(-1, start=55, length=3) # dx, dy, dz between green and red
        obs_body = self.observation.narrow(-1, start=0, length=self.dim_body)
        obs_objects = [torch.cat((torch.cat(obs_body.shape[0] * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.num_blocks),
                                  self.observation.narrow(-1, start=self.dim_object*i + self.dim_body, length=self.dim_object)),
                                 dim=-1) for i in range(self.num_blocks)]

        if self.symmetry_trick:
            all_inputs = []
            for i in range(self.num_blocks):
                for j in range(self.num_blocks):
                    if i < j:
                        all_inputs.append(torch.cat([ag[:, self.first_inds], obs_body, self.g[:, self.first_inds], obs_objects[i], obs_objects[j]], dim=1))
                    elif j < i:
                        all_inputs.append(torch.cat([ag[:, self.second_inds], obs_body, self.g[:, self.second_inds], obs_objects[i], obs_objects[j]], dim=1))

            input_actor = torch.stack(all_inputs)

        elif self.combinations_trick:
            # Get indexes of atomic goals and corresponding object tuple
            extractors = [torch.zeros((self.anchor_g.shape[1], 1)) for _ in range(self.anchor_g.shape[1])]
            for i in range(len(extractors)):
                extractors[i][i, :] = 1.

            # The trick is to create selector matrices that, when multiplied with goals retrieves certain bits. Then the sign of the difference
            # between bits gives which objet goes above the the other

            idxs_bits = [torch.empty(self.anchor_g.shape[0], 2) for _ in range(3)]
            idxs_objects = [torch.empty(self.anchor_g.shape[0], 2) for _ in range(3)]

            for i, ((o1, o2), (j, k)) in enumerate(zip([(0, 1), (0, 2), (1, 2)], [(3, 5), (4, 7), (6, 8)])):
                stacked = torch.cat([extractors[j], extractors[k]], dim=1)
                multiplied_matrix = torch.matmul(self.anchor_g, stacked)
                selector = multiplied_matrix[:, 0] - multiplied_matrix[:, 1]

                idxs_bits[i] = torch.tensor([i, k]).repeat(self.anchor_g.shape[0], 1).long()
                idxs_bits[i][selector >= 0] = torch.Tensor([i, j]).long()

                idxs_objects[i] = torch.tensor([o2, o1]).repeat(self.anchor_g.shape[0], 1).long()
                idxs_objects[i][selector >= 0] = torch.Tensor([o1, o2]).long()

            # Gather 2 bits achieved goal
            ag_1_2 = self.ag.gather(1, idxs_bits[0])
            ag_1_3 = self.ag.gather(1, idxs_bits[1])
            ag_2_3 = self.ag.gather(1, idxs_bits[2])

            # Gather 2 bits goal
            g_1_2 = self.g.gather(1, idxs_bits[0])
            g_1_3 = self.g.gather(1, idxs_bits[1])
            g_2_3 = self.g.gather(1, idxs_bits[2])

            obs_object_tensor = torch.stack(obs_objects)

            obs_objects_pairs_list = []
            for idxs_objects in idxs_objects:
                permuted_idxs = idxs_objects.unsqueeze(0).permute(2, 1, 0)
                permuted_idxs = permuted_idxs.repeat(1, 1, obs_object_tensor.shape[2])
                obs_objects_pair = obs_object_tensor.gather(0, permuted_idxs)
                obs_objects_pairs_list.append(obs_objects_pair)

            input_1_2 = torch.cat([ag_1_2, torch.cat([g_1_2, obs_body], dim=1), obs_objects_pairs_list[0][0, :, :],
                                   obs_objects_pairs_list[0][1, :, :]], dim=1)
            input_1_3 = torch.cat([ag_1_3, torch.cat([g_1_3, obs_body], dim=1), obs_objects_pairs_list[1][0, :, :],
                                   obs_objects_pairs_list[1][1, :, :]], dim=1)
            input_2_3 = torch.cat([ag_2_3, torch.cat([g_2_3, obs_body], dim=1), obs_objects_pairs_list[2][0, :, :],
                                   obs_objects_pairs_list[2][1, :, :]], dim=1)

            input_actor = torch.stack([input_1_2, input_1_3, input_2_3])

        else:
            if self.language:
                body_input_actor = torch.cat([goal_embeddings, obs_body], dim=1)
            else:
                body_input_actor = torch.cat([self.g, obs_body], dim=1)
            obj_input_actor = [obs_objects[i] for i in range(self.num_blocks)]

            # Parallelization by stacking input tensors

            if self.continuous_trick:
                all_inputs = []
                for i in range(self.num_blocks):
                    for j in range(self.num_blocks):
                        if i != j:
                            all_inputs.append(
                                torch.cat([ag[:, self.obj_ids[i]], ag[:, self.obj_ids[j]], obs_body, self.g[:, self.obj_ids[i]],
                                           self.g[:, self.obj_ids[j]], obs_objects[i], obs_objects[j]], dim=1))

                input_actor = torch.stack(all_inputs)
            else:
                if not self.include_ag:
                    input_actor = torch.stack([torch.cat([body_input_actor, x[0], x[1], obs_distances], dim=1) for x in permutations(obj_input_actor, 2)])
                    # input_actor = torch.stack([torch.cat([body_input_actor, obj_input_actor[0], obj_input_actor[1],
                    #                                       obs_distances], dim=1)])
                else:
                    input_actor = torch.stack([torch.cat([ag, body_input_actor, x[0], x[1]], dim=1) for x in permutations(obj_input_actor, 2)])

        output_phi_actor = self.single_phi_actor(input_actor).sum(dim=0)
        # self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)
        else:
            _, self.log_prob, self.pi_tensor = self.rho_actor.sample(output_phi_actor)

    def forward_pass(self, obs, ag, g, anchor_g=None, eval=False, actions=None, language_goals=None):
        batch_size = obs.shape[0]
        self.observation = obs
        self.ag = ag
        self.g = g

        self.anchor_g = anchor_g

        if self.language:
            if ONE_HOT:
                goal_embeddings = torch.tensor(np.array([self.simple_encoder[lg] for lg in language_goals]), dtype=torch.float32)
            else:
                encodings = np.array([self.one_hot_language[lg] for lg in language_goals])
                # old
                # encodings = np.array([self.one_hot_language[str(sg)] for sg in g])
                encodings = torch.tensor(encodings, dtype=torch.float32)
                if UNIQUE_ENCODER:
                    goal_embeddings = self.critic_sentence_encoder.forward(encodings)[0][:, -1, :]
                    with torch.no_grad():
                        goal_embeddings_target = self.target_critic_sentence_encoder.forward(encodings)[0][:, -1, :]
                else:
                    goal_embeddings = self.policy_sentence_encoder.forward(encodings)[0][:, -1, :]

        obs_distances = self.observation.narrow(-1, start=55, length=3)
        obs_body = self.observation[:, :self.dim_body]
        obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.num_blocks),
                               obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
                              for i in range(self.num_blocks)]

        if self.symmetry_trick:
            all_inputs = []
            for i in range(self.num_blocks):
                for j in range(self.num_blocks):
                    if i < j:
                        all_inputs.append(torch.cat([ag[:, self.first_inds], obs_body, self.g[:, self.first_inds], obs_objects[i], obs_objects[j]], dim=1))
                    elif j < i:
                        all_inputs.append(torch.cat([ag[:, self.second_inds], obs_body, self.g[:, self.second_inds], obs_objects[i], obs_objects[j]], dim=1))

            input_actor = torch.stack(all_inputs)

        elif self.combinations_trick:
            # Get indexes of atomic goals and corresponding object tuple
            extractors = [torch.zeros((self.anchor_g.shape[1], 1)) for _ in range(self.anchor_g.shape[1])]
            for i in range(len(extractors)):
                extractors[i][i, :] = 1.

            # The trick is to create selector matrices that, when multiplied with goals retrieves certain bits. Then the sign of the difference
            # between bits gives which objet goes above the the other

            idxs_bits = [torch.empty(self.anchor_g.shape[0], 2) for _ in range(3)]
            idxs_objects = [torch.empty(self.anchor_g.shape[0], 2) for _ in range(3)]

            for i, ((o1, o2), (j, k)) in enumerate(zip([(0, 1), (0, 2), (1, 2)], [(3, 5), (4, 7), (6, 8)])):
                stacked = torch.cat([extractors[j], extractors[k]], dim=1)
                multiplied_matrix = torch.matmul(self.anchor_g, stacked.double())
                selector = multiplied_matrix[:, 0] - multiplied_matrix[:, 1]

                idxs_bits[i] = torch.tensor([i, k]).repeat(self.anchor_g.shape[0], 1).long()
                idxs_bits[i][selector >= 0] = torch.Tensor([i, j]).long()

                idxs_objects[i] = torch.tensor([o2, o1]).repeat(self.anchor_g.shape[0], 1).long()
                idxs_objects[i][selector >= 0] = torch.Tensor([o1, o2]).long()

            # Gather 2 bits achieved goal
            ag_1_2 = self.ag.gather(1, idxs_bits[0])
            ag_1_3 = self.ag.gather(1, idxs_bits[1])
            ag_2_3 = self.ag.gather(1, idxs_bits[2])

            # Gather 2 bits goal
            g_1_2 = self.g.gather(1, idxs_bits[0])
            g_1_3 = self.g.gather(1, idxs_bits[1])
            g_2_3 = self.g.gather(1, idxs_bits[2])

            obs_object_tensor = torch.stack(obs_objects)

            obs_objects_pairs_list = []
            for idxs_objects in idxs_objects:
                permuted_idxs = idxs_objects.unsqueeze(0).permute(2, 1, 0)
                permuted_idxs = permuted_idxs.repeat(1, 1, obs_object_tensor.shape[2])
                obs_objects_pair = obs_object_tensor.gather(0, permuted_idxs)
                obs_objects_pairs_list.append(obs_objects_pair)

            input_1_2 = torch.cat([ag_1_2, torch.cat([g_1_2, obs_body], dim=1), obs_objects_pairs_list[0][0, :, :],
                                   obs_objects_pairs_list[0][1, :, :]], dim=1)
            input_1_3 = torch.cat([ag_1_3, torch.cat([g_1_3, obs_body], dim=1), obs_objects_pairs_list[1][0, :, :],
                                   obs_objects_pairs_list[1][1, :, :]], dim=1)
            input_2_3 = torch.cat([ag_2_3, torch.cat([g_2_3, obs_body], dim=1), obs_objects_pairs_list[2][0, :, :],
                                   obs_objects_pairs_list[2][1, :, :]], dim=1)

            input_actor = torch.stack([input_1_2, input_1_3, input_2_3])

        else:
            if self.language:
                body_input = torch.cat([goal_embeddings, obs_body], dim=1)
                body_input_target = torch.cat([goal_embeddings_target, obs_body], dim=1)
            else:
                body_input = torch.cat([self.g, obs_body], dim=1)
            obj_input = [obs_objects[i] for i in range(self.num_blocks)]

            # Parallelization by stacking input tensors
            if not self.include_ag:
                input_actor = torch.stack([torch.cat([body_input, x[0], x[1], obs_distances], dim=1) for x in permutations(obj_input, 2)])
                input_actor_target = torch.stack([torch.cat([body_input_target, x[0], x[1],
                                                             obs_distances], dim=1) for x in permutations(obj_input, 2)])
                # input_actor = torch.stack([torch.cat([body_input, obs_objects[0], obs_objects[1], obs_distances], dim=1)])
                # input_actor_target = torch.stack([torch.cat([body_input_target, obs_objects[0], obs_objects[1], obs_distances], dim=1)])
            else:
                input_actor = torch.stack([torch.cat([ag, body_input, x[0], x[1]], dim=1) for x in permutations(obj_input, 2)])

        output_phi_actor = self.single_phi_actor(input_actor).sum(dim=0)
        if not eval:
            self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)
        else:
            _, self.log_prob, self.pi_tensor = self.rho_actor.sample(output_phi_actor)

        # The critic part
        repeat_pol_actions = self.pi_tensor.repeat(self.n_permutations, 1, 1)
        input_critic = torch.cat([input_actor, repeat_pol_actions], dim=-1)
        input_critic_target = torch.cat([input_actor_target, repeat_pol_actions], dim=-1)
        if actions is not None:
            repeat_actions = actions.repeat(self.n_permutations, 1, 1)
            input_critic_with_act = torch.cat([input_actor, repeat_actions], dim=-1)
            input_target_critic_with_act = torch.cat([input_actor_target, repeat_actions], dim=-1)
            input_critic = torch.cat([input_critic, input_critic_with_act], dim=0)
            input_critic_target = torch.cat([input_critic_target, input_target_critic_with_act], dim=0)

        with torch.no_grad():
            output_phi_target_critic_1, output_phi_target_critic_2 = self.single_phi_target_critic(input_critic_target[:self.n_permutations])
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




