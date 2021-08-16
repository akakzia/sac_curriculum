import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from time import ctime
import pickle

from proximal_generator.arguments import get_args
from utils import load_dataset, split_data
from models import ProximalGoalGenerator

def launch(args):
    all_results = []
    for i in range(args.num_seeds):
        print('\nTraining model for seed {} / {} ...'.format(i + 1, args.num_seeds))
        init_configs, init_states, final_configs, final_states, init_to_finals, device, data_loader, set_ids = process_data(args)

        results = train(init_configs, init_states, final_configs, final_states, init_to_finals, device, data_loader, set_ids, args, i)

        all_results.append(results)

        args.seed = np.random.randint(1e6)

    all_results = np.array(all_results)
    print(np.mean(all_results, axis=0))
    print(np.std(all_results, axis=0))

def process_data(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_configs, init_states, final_configs, final_states, init_to_finals = load_dataset()
    print('Size of the dataset: ', init_configs.shape[0])
    # construct the different test sets
    set_ids, dataset = split_data(init_configs, init_states, final_configs, init_to_finals)

    for i, s in enumerate(set_ids):
        print('Len Set ', i, ': ', len(s))

    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    return init_configs, init_states, final_configs, final_states, init_to_finals, device, data_loader, set_ids

def train(init_configs, init_states, final_configs, final_states, init_to_finals, device, data_loader, set_ids, args, vae_id):
    model = ProximalGoalGenerator(init_configs, data_loader,  device, args)

    for epoch in range(args.epochs + 1):
        model.train()

    # if args.save_model:
    model.save(vae_id)

    results = model.evaluate(init_configs, init_states, final_configs, set_ids, init_to_finals)

    return results.copy()
    #
    # with open(args.save_path + 'res{}.pkl'.format(vae_id), 'wb') as f:
    #     pickle.dump(results, f)
    # return results.copy()

if __name__ == '__main__':
    # Get parameters
    args = get_args()

    args.save_path = os.path.join(os.getcwd(), 'proximal_generator', args.save_dir)
    args.data_path = os.path.join(os.getcwd(), 'data', args.data_name + '.pkl')
    print('[{}] Launching Language Goal Generator training'.format(ctime()))
    print('Relational Generator: {}'.format(args.relational))

    launch(args)