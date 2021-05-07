import torch
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

def get_ltor_masks_and_position_ids(data, eod_token):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)
    # attention_mask = torch.zeros(
    #     (seq_length, divide(args.num_attention_heads, 2), batch_size, batch_size), device=data.device
    # )

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids
