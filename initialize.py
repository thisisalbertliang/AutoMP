import torch
import torch.distributed
import argparse


def init_distributed():
    assert torch.cuda.is_available(), 'AutoMP requires CUDA'
    assert not torch.distributed.is_initialized(), 'torch distributed is already initialized, should not initialize again'

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)

    if args.local_rank == 0:
        print('> initializing torch distributed ...', flush=True)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

# def init_model_parallel


# if __name__ == '__main__':
#     init_distributed()
