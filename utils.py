from arguments import get_args

def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    assert numerator % denominator == 0, f'{numerator} is not divisible by {denominator}'
    return numerator // denominator

def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank):
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f, index_l

def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size, rank
    )

def print_rank_0(msg: str):
    args = get_args()
    if args.local_rank == 0:
        print(msg)
