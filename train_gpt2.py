import torch
import torch.distributed
import numpy as np
import time
from initialize import init_distributed
from arguments import parse_args, get_args
from utils import print_rank_0, get_ltor_masks_and_position_ids
from model.gpt2 import GPT2


def train():
    
    # Initialize torch.distributed
    init_distributed()

    print_rank_0('AutoMP: training GPT2...')

    batch_size = 8
    sequence_length = 256#1024
    hidden_size = 512#2048
    vocab_size = 1024#4096
    attention_dropout = hidden_dropout = 0.1
    num_layers = 2
    layernorm_epsilon = 1e-5
    num_attention_heads = 2

    input_indices = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
    input_indices = input_indices.to(torch.cuda.current_device())
    position_indices = torch.tile(torch.arange(start=0, end=sequence_length), (batch_size, 1))
    position_indices = position_indices.to(torch.cuda.current_device())
    
    def init_method_normal(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
    def gpt2_attention_mask_func(attention_scores, ltor_mask):
        attention_scores.masked_fill_(ltor_mask, -10000.0)
        return attention_scores

    gpt2 = GPT2(
        hidden_size, vocab_size, sequence_length, hidden_dropout,
        gpt2_attention_mask_func, num_layers, layernorm_epsilon, 
        num_attention_heads, attention_dropout,
        init_method_normal,
    )

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(input_indices, vocab_size - 1)
    gpt2.forward(input_indices, position_indices, attention_mask)
    

if __name__ == '__main__':
    # Parse command line arguments
    parse_args()

    args = get_args()

    train()