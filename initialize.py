import torch
import torch.distributed
from arguments import get_args
from utils import print_rank_0


def init_distributed():
    assert torch.cuda.is_available(), 'AutoMP requires CUDA'
    assert not torch.distributed.is_initialized(), 'torch distributed is already initialized, should not initialize again'

    args = get_args()

    torch.cuda.set_device(args.local_rank)

    print_rank_0('AutoMP: initializing torch distributed ...')
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    print_rank_0(f'AutoMP: Successfully initialized torch distributed with world-size={torch.distributed.get_world_size()}')

# def init_model_parallel


# if __name__ == '__main__':
#     init_distributed()
