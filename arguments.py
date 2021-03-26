import argparse
import os

GLOBAL_ARGS = None


def parse_args():
    parser = argparse.ArgumentParser(description='AutoMP arguments')

    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--model-parallel-size', type=int, default=1,
                        help='number of model parallel processes')
    # Add future AutoMP arguments here...

    # Parse
    global GLOBAL_ARGS
    GLOBAL_ARGS = parser.parse_args()


def get_args():
    assert GLOBAL_ARGS is not None
    return GLOBAL_ARGS