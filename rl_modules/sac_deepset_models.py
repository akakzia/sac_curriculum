import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import permutations
import numpy as np
from language.utils import OneHotEncoder, analyze_inst, Vocab
from mpi4py import MPI
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

ONE_HOT = False

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
    def __init__(self, inp, out):
        super(RhoActor, self).__init__()
        self.linear1 = nn.Linear(inp, 256)
        self.mean_linear = nn.Linear(256, out)
        self.log_std_linear = nn.Linear(256, out)

        self.apply(weights_init_)


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

    def forward(self, inp1, inp2):
        x1 = F.relu(self.linear1(inp1))
        x1 = F.relu(self.linear2(x1))

        x2 = F.relu(self.linear4(inp2))
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

class Critic(nn.Module):
    def __init__(self, nb_objects,  dim_phi_critic_input, dim_phi_critic_output,
                 dim_rho_critic_input, dim_rho_critic_output, one_hot_language,
                 vocab_size, embedding_size, nb_permutations, dim_body, dim_object):
        super(Critic, self).__init__()
        self.critic_sentence_encoder1 = nn.LSTM(input_size=vocab_size,
                                               hidden_size=embedding_size,
                                               batch_first=True)
        self.critic_sentence_encoder2 = nn.LSTM(input_size=vocab_size,
                                                hidden_size=embedding_size,
                                                batch_first=True)

        self.nb_permutations = nb_permutations
        self.one_hot_language = one_hot_language
        self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]
        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object
        self.single_phi_critic = SinglePhiCritic(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_critic = RhoCritic(dim_rho_critic_input, dim_rho_critic_output)

    def forward(self, obs, act, language_goals):

        batch_size = obs.shape[0]
        assert batch_size == len(language_goals)

        # encode language goals
        encodings = torch.tensor(np.array([self.one_hot_language[lg] for lg in language_goals]), dtype=torch.float32)
        l_emb1 = self.critic_sentence_encoder1.forward(encodings)[0][:, -1, :]
        l_emb2 = self.critic_sentence_encoder2.forward(encodings)[0][:, -1, :]


        obs_body = obs[:, :self.dim_body]
        obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
                                  obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
                       for i in range(self.nb_objects)]

        # Parallelization by stacking input tensors
        inp1 = torch.stack([torch.cat([l_emb1, obs_body, act, x[0], x[1]], dim=1) for x in permutations(obs_objects, 2)])
        inp2 = torch.stack([torch.cat([l_emb2, obs_body, act, x[0], x[1]], dim=1) for x in permutations(obs_objects, 2)])


        output_phi_critic_1, output_phi_critic_2 = self.single_phi_critic(inp1, inp2)
        output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
        output_phi_critic_2 = output_phi_critic_2.sum(dim=0)
        q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
        return q1_pi_tensor, q2_pi_tensor

    def encode_language(self, language_goals):
        if isinstance(language_goals, str):
            encodings = torch.tensor(self.one_hot_language[language_goals], dtype=torch.float32).unsqueeze(dim=0)
        else:
            encodings = torch.tensor(np.array([self.one_hot_language[lg] for lg in language_goals]), dtype=torch.float32)
        l_emb1 = self.critic_sentence_encoder1.forward(encodings)[0][:, -1, :]
        return l_emb1

class Actor(nn.Module):
    def __init__(self, nb_objects,  dim_phi_actor_input, dim_phi_actor_output,
                 dim_rho_actor_input, dim_rho_actor_output, dim_body, dim_object):
        super(Actor, self).__init__()


        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object
        self.single_phi_actor = SinglePhiActor(dim_phi_actor_input, 256, dim_phi_actor_output)
        self.rho_actor = RhoActor(dim_rho_actor_input, dim_rho_actor_output)
        self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]


    def forward(self, obs, l_emb):
        batch_size = obs.shape[0]
        assert batch_size == l_emb.shape[0]


        obs_body = obs[:, :self.dim_body]
        obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
                                  obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
                       for i in range(self.nb_objects)]

        # Parallelization by stacking input tensors
        inp = torch.stack([torch.cat([l_emb, obs_body, x[0], x[1]], dim=1) for x in permutations(obs_objects, 2)])


        output_phi_actor = self.single_phi_actor(inp)
        output_phi_actor = output_phi_actor.sum(dim=0)
        mean, logstd = self.rho_actor(output_phi_actor)
        return mean, logstd

    def sample(self, obs, l_emb):
        mean, log_std = self.forward(obs, l_emb)
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


class DeepSetSAC:
    def __init__(self, instructions, env_params, args):
        # A raw version of DeepSet-based SAC without attention mechanism
        self.observation = None

        self.dim_body = 10
        self.dim_object = 15
        self.dim_act = env_params['action']
        self.num_blocks = 3
        self.n_permutations = len([x for x in permutations(range(self.num_blocks), 2)])

        self.instructions = instructions
        self.nb_instructions = len(self.instructions)
        self.embedding_size = self.nb_instructions if ONE_HOT else args.embedding_size

        split_instructions, max_seq_length, word_set = analyze_inst(self.instructions)
        vocab = Vocab(word_set)
        self.one_hot_encoder = OneHotEncoder(vocab, max_seq_length)
        self.one_hot_language = dict(zip(self.instructions, [self.one_hot_encoder.encode(s) for s in split_instructions]))

        # asserts all cpu have same encodings
        all_encodings = MPI.COMM_WORLD.gather(self.one_hot_encoder.encode(['put', 'green', 'close_to', 'blue']), root=0)
        if MPI.COMM_WORLD.Get_rank() == 0:
            for i in range(len(all_encodings)):
                for j in range(i, len(all_encodings)):
                    assert np.all(np.array(all_encodings[i]) == np.array(all_encodings[j]))
            print('ALRIGHT0')
        # asserts all cpu have same encodings
        all_encodings = MPI.COMM_WORLD.gather(self.one_hot_language['Put green close_to blue'], root=0)
        if MPI.COMM_WORLD.Get_rank() == 0:
            for i in range(len(all_encodings)):
                for j in range(i, len(all_encodings)):
                    assert np.all(np.array(all_encodings[i]) == np.array(all_encodings[j]))
            print('ALRIGHT1')

        # double_critic_attention = double_critic_attention

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        # Define dimensions

        dim_input_objects = 2 * (self.num_blocks + self.dim_object)
        dim_phi_actor_input = self.embedding_size + self.dim_body + dim_input_objects
        dim_phi_actor_output = 3 * dim_phi_actor_input
        dim_rho_actor_input = dim_phi_actor_output
        dim_rho_actor_output = self.dim_act

        dim_phi_critic_input = dim_phi_actor_input + self.dim_act
        dim_phi_critic_output = 3 * dim_phi_critic_input
        dim_rho_critic_input = dim_phi_critic_output
        dim_rho_critic_output = 1


        self.critic = Critic(self.num_blocks, dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output,
                             self.one_hot_language, vocab.size, self.embedding_size, self.n_permutations, self.dim_body, self.dim_object)
        self.critic_target = Critic(self.num_blocks, dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output,
                             self.one_hot_language, vocab.size, self.embedding_size, self.n_permutations, self.dim_body, self.dim_object)
        self.actor = Actor(self.num_blocks, dim_phi_actor_input, dim_phi_actor_output, dim_rho_actor_input, dim_rho_actor_output, self.dim_body, self.dim_object)

    def policy_forward_pass(self, obs, no_noise=False, language_goal=None):

        l_emb = self.critic.encode_language(language_goal)
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, l_emb)
        else:
            _, self.log_prob, self.pi_tensor = self.actor.sample(obs, l_emb)

    def forward_pass(self, obs, actions=None, language_goals=None):

        l_emb = self.critic.encode_language(language_goals)
        self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, l_emb)

        if actions is not None:
            self.q1_pi_tensor, self.q2_pi_tensor = self.critic.forward(obs, self.pi_tensor, language_goals)
            return self.critic.forward(obs, actions, language_goals)
        else:
            with torch.no_grad():
                self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.critic.forward(obs, self.pi_tensor, language_goals)
            self.q1_pi_tensor, self.q2_pi_tensor = None, None





