import torch
import numpy as np
import torch.nn.functional as F
import pickle as pkl
from rl_modules.networks import QNetworkFlat, GaussianPolicyFlat
from rl_modules.rl_agent import hard_update


# Hyper-parameters
lr_actor = 0.003
lr_critic = 0.003
lr_entropy = 0.003
batch_size = 256
alpha = 0.02
gamma = 0.98
polyak = 0.95

path = '/home/ahmed/Documents/DECSTR/sac_curriculum/planning/data/'
file_name = 'BtoL.pkl'

nb_objects = 3
state_dim = nb_objects * (nb_objects - 1) * 3 // 2
action_dim = state_dim
size = 10000
freq_target_update = 2


env_params = {'obs': 0, 'goal': state_dim, 'action': action_dim}

# Models definition
actor_network = GaussianPolicyFlat(env_params)
critic_network = QNetworkFlat(env_params)
critic_target_network = QNetworkFlat(env_params)
hard_update(critic_target_network, critic_network)
# Optimizer
policy_optim = torch.optim.Adam(actor_network.parameters(), lr=lr_actor)
critic_optim = torch.optim.Adam(critic_network.parameters(), lr=lr_critic)

target_entropy = -torch.prod(torch.Tensor(action_dim)).item()
log_alpha = torch.zeros(1, requires_grad=True)
alpha_optim = torch.optim.Adam([log_alpha], lr=lr_entropy)


with open(path + file_name, 'rb') as f:
    data = pkl.load(f)

buffer = {'c': np.empty([size, state_dim]),
          'c_next': np.empty([size, state_dim]),
          'c_g': np.empty([size, state_dim]),
          'actions': np.empty([size, state_dim]),
          'rewards': np.empty([size, 1]),
         }

buffer_current_size = 0

actions_set = set()

for e in data:
    T = len(e['obs'])
    for i in range(T-1):
        if str(e['ag'][i]) != str(e['ag'][i+1]):
            buffer['c'][buffer_current_size] = np.clip(e['ag'][i] + 1, 0, 1)
            buffer['c_next'][buffer_current_size] = np.clip(e['ag'][i+1] + 1, 0, 1)
            action = np.clip(e['ag'][i+1] + 1, 0, 1) - np.clip(e['ag'][i] + 1, 0, 1)
            actions_set.add(str(action))
            buffer['actions'][buffer_current_size] = action
            buffer['c_g'][buffer_current_size] = np.clip(e['g'][i] + 1, 0, 1)
            buffer['rewards'][buffer_current_size] = 1 if str(e['ag'][i+1]) == str(e['g'][i]) else 0
            buffer_current_size += 1

print('Stored {} transitions in buffer.'.format(buffer_current_size))
print('There are {} actions'.format(len(actions_set)))
stop = 1


def soft_update_target_network(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - polyak) * param.data + polyak * target_param.data)


def update_flat(ag, g, ag_next, actions, rewards):
    inputs_norm = np.concatenate([ag, g], axis=1)
    inputs_next_norm = np.concatenate([ag_next, g], axis=1)

    # Tensorize
    inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
    inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    r_tensor = torch.tensor(rewards, dtype=torch.float32).reshape(rewards.shape[0], 1)

    with torch.no_grad():
        actions_next, log_pi_next, _ = actor_network.sample(inputs_next_norm_tensor)
        qf1_next_target, qf2_next_target = critic_target_network(inputs_next_norm_tensor, actions_next)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * log_pi_next
        next_q_value = r_tensor + gamma * min_qf_next_target

    # the q loss
    qf1, qf2 = critic_network(inputs_norm_tensor, actions_tensor)
    qf1_loss = F.mse_loss(qf1, next_q_value)
    qf2_loss = F.mse_loss(qf2, next_q_value)

    # the actor loss
    pi, log_pi, _ = actor_network.sample(inputs_norm_tensor)
    qf1_pi, qf2_pi = critic_network(inputs_norm_tensor, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)
    policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

    # start to update the network
    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    # update the critic_network
    critic_optim.zero_grad()
    qf1_loss.backward()
    critic_optim.step()

    critic_optim.zero_grad()
    qf2_loss.backward()
    critic_optim.step()

    alpha_loss = torch.tensor(0.)
    alpha_tlogs = torch.tensor(alpha)

    return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


def update_network():
    transitions = {}
    ids = np.random.randint(buffer_current_size, size=batch_size)
    for key in buffer.keys():
        transitions[key] = buffer[key][ids]

    c, c_g, c_next, actions, rewards = transitions['c'], transitions['c_g'], transitions['c_next'], transitions['actions'], transitions['rewards']

    critic_1_loss, critic_2_loss, actor_loss, _, _ = update_flat(c, c_g, c_next, actions, rewards)

    print('Critic 1 loss: {} | Critic 2 loss: {} | Actor loss: {}'.format(critic_1_loss, critic_2_loss, actor_loss))


total_iter = 0
for i in range(1):
    torch.autograd.set_detect_anomaly(True)
    total_iter += 1
    update_network()

    if total_iter % freq_target_update == 0:
        soft_update_target_network(critic_target_network, critic_network)