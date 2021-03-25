import torch
import torch.distributed
import numpy as np
from initialize import init_distributed
from arguments import parse_args, get_args
from model.linear import ParallelLinear
from utils import print_rank_0


def train():
    # Parse command line arguments
    parse_args()
    # Initialize torch.distributed
    init_distributed()

    print_rank_0('AutoMP: training MLP...')
    # Use MNIST data
    train_data = np.genfromtxt('data/digitstrain.txt', delimiter=",")
    train_X = torch.tensor(train_data[:, :-1], dtype=torch.float, device=torch.cuda.current_device())
    train_Y = torch.tensor(train_data[:, -1], dtype=torch.float, device=torch.cuda.current_device())
    print_rank_0(f'AutoMP: number of training samples: {train_X.size()[0]}')
    print_rank_0(f'AutoMP: number of features: {train_X.size()[1]}')

    num_features = train_X.size()[1]
    num_hidden = 1024
    parallel_linear = ParallelLinear(num_features, num_hidden)
    print_rank_0('AutoMP: Successfully initialized ParallelLinear')

    output = parallel_linear.forward(train_X)
    # take a look at this shit
    print_rank_0(str(output))
    print_rank_0(str(output.shape))


if __name__ == '__main__':
    train()