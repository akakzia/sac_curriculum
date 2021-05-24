import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetworkFlat(nn.Module):
    def __init__(self, env_params):
        super(QNetworkFlat, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(env_params['obs'] + 2 * env_params['goal'] + env_params['action'], 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(env_params['obs'] + 2 * env_params['goal'] + env_params['action'], 256)
        self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicyFlat(nn.Module):
    def __init__(self, env_params, action_space=None):
        super(GaussianPolicyFlat, self).__init__()

        self.linear1 = nn.Linear(env_params['obs'] + 2 * env_params['goal'], 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, env_params['action'])
        self.log_std_linear = nn.Linear(256, env_params['action'])

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
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
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyFlat, self).to(device)


# DeepSet networks. Phi designs pre-aggregation networks whereas Rho designs post-aggregation networks
class PhiActorDeepSet(nn.Module):
    def __init__(self, inp, hid, out):
        super(PhiActorDeepSet, self).__init__()
        self.linear1 = nn.Linear(inp, hid)
        self.linear2 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = F.relu(self.linear1(inp))
        x = F.relu(self.linear2(x))

        return x


class RhoActorDeepSet(nn.Module):
    def __init__(self, inp, out, action_space=None):
        super(RhoActorDeepSet, self).__init__()
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

    def forward(self, state):
        x = F.relu(self.linear1(state))
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


class PhiCriticDeepSet(nn.Module):
    def __init__(self, inp, hid, out):
        super(PhiCriticDeepSet, self).__init__()
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


class RhoCriticDeepSet(nn.Module):
    def __init__(self, inp, out):
        super(RhoCriticDeepSet, self).__init__()
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


class GnnAttention(nn.Module):
    def __init__(self, inp, hid, out):
        super(GnnAttention, self).__init__()
        self.linear1 = nn.Linear(inp, hid)
        self.linear2 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = F.relu(self.linear1(inp))
        x = F.gumbel_softmax(self.linear2(x), hard=True)

        return x


class GnnMessagePassing(nn.Module):
    def __init__(self, inp, out):
        super(GnnMessagePassing, self).__init__()
        self.linear1 = nn.Linear(inp, 256)
        self.linear2 = nn.Linear(256, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = F.relu(self.linear1(inp))
        x = F.relu(self.linear2(x))

        return x


class GATLayer(nn.Module):
    """
    Simple PyTorch Implementation of the Graph Attention layer.
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout  # drop prob = 0.6
        self.in_features = in_features  #
        self.out_features = out_features  #
        self.alpha = alpha  # LeakyReLU with negative input slope, alpha = 0.2
        self.concat = concat  # conacat = True for all layers except the output layer.

        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features + 2, 1)))  # +2 to add goal
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, ag, g, predicate_ids, edges, n_permutations):
        batch_size = input.size()[0]
        # Linear Transformation
        h = torch.matmul(input, self.W)
        N = h.size()[1]

        # Attention Mechanism
        # a_input = torch.cat([h.repeat(1, 1, N).view(batch_size, N * N, -1), h.repeat(1, N, 1)], dim=-1).view(batch_size, N, -1, 2 * self.out_features)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        a_input = torch.stack([torch.cat([g[:, predicate_ids[i]], h[:, edges[i][0], :], h[:, edges[i][1], :]], dim=-1)
                         for i in range(n_permutations)], dim=1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        return attention, a_input
        # h_prime = torch.matmul(attention, h)
        #
        # if self.concat:
        #     return F.elu(h_prime)
        # else:
        #     return h_prime
