import torch
import torch.distributed
import numpy as np
import time
from initialize import init_distributed
from arguments import parse_args, get_args
from model.linear import ParallelLinear
from utils import print_rank_0
from model.cross_entropy import parallel_cross_entropy
from model.mlp import ParallelMLP


def train():
    
    # Initialize torch.distributed
    init_distributed()

    print_rank_0('AutoMP: training GPT2...')
    # Use fake train data
    batch_size = 8
    sequence_length = 1024
    train_X = torch.normal(mean=0.0, std=1.0, size=(batch_size, sequence_length))
    print_rank_0(f'train_X shape: {train_X.size()}')

    # TBC ...

if __name__ == '__main__':
    # Parse command line arguments
    parse_args()

    args = get_args()

    train()