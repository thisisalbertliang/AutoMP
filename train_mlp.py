import torch
import torch.distributed
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
    num_train_samples = 100
    num_features = 784
    # Use some fake shit as train data for now
    train_X = torch.rand(num_train_samples, num_features).to(device=torch.cuda.current_device())

    num_hidden = 1024
    parallel_linear = ParallelLinear(num_features, num_hidden)
    print_rank_0('AutoMP: Successfully initialized ParallelLinear')

    output = parallel_linear.forward(train_X)
    # take a look at this shit
    print_rank_0(str(output))


if __name__ == '__main__':
    train()