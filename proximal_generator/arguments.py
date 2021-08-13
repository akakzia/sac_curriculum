import argparse
import numpy as np


"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--relational", default=False, help='Use a GNN based model')
    # Training parameters
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--encoder-layer-sizes", type=list, default=[128, 128])
    parser.add_argument("--decoder-layer-sizes", type=list, default=[128, 128])
    parser.add_argument("--latent-size", type=int, default=17)

    parser.add_argument("--k-param", type=int, default=0.01)

    parser.add_argument("--save-dir", type=str, default='data/')
    parser.add_argument("--save-model", action='store_true')
    parser.add_argument("--data-name", type=str, default='trajectories_5blocks')

    args = parser.parse_args()

    return args
