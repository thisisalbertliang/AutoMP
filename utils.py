from arguments import get_args

def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    assert numerator % denominator == 0, f'{numerator} is not divisible by {denominator}'
    return numerator // denominator


def print_rank_0(msg: str):
    args = get_args()
    if args.local_rank == 0:
        print(msg)
