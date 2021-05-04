import argparse
import os

GLOBAL_ARGS = None


def parse_args():
    parser = argparse.ArgumentParser(description='AutoMP arguments')

    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--model-parallel-size', type=int, default=1,
                        help='number of model parallel processes')
    # Add future AutoMP arguments here...

    parser.add_argument('--exp-name', type=str, default='exp')
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--sequence-length', type=int, default=1024)
    parser.add_argument('--hidden-dropout', type=float, default=.1)
    parser.add_argument('--attention-dropout', type=float, default=.1)
    parser.add_argument('--vocab-size', type=int, default=4096)
    parser.add_argument('--num-attention-heads', type=int, default=4)
    parser.add_argument('--layernorm-epsilon', type=float, default=1e-5, help='Layer norm epsilon.')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of transformer layers')

    # Parse
    global GLOBAL_ARGS
    GLOBAL_ARGS = parser.parse_args()


def get_args():
    assert GLOBAL_ARGS is not None
    return GLOBAL_ARGS
