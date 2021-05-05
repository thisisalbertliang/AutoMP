import torch
import torch.distributed
import numpy as np
import time
from initialize import init_distributed
from arguments import parse_args, get_args
from utils import print_rank_0, get_ltor_masks_and_position_ids
from model.gpt2 import GPT2
from profiler import Profiler
import os


def train():
    
    # Initialize torch.distributed
    init_distributed()

    print_rank_0('AutoMP: training GPT2...')

    batch_size = args.batch_size 
    sequence_length = args.sequence_length
    hidden_size = args.hidden_size
    vocab_size = args.vocab_size
    hidden_dropout = args.hidden_dropout
    attention_dropout = args.attention_dropout
    num_layers = args.num_layers
    layernorm_epsilon = args.layernorm_epsilon
    num_attention_heads = args.num_attention_heads

    input_indices = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
    input_indices = input_indices.to(torch.cuda.current_device())
    labels = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
    labels = labels.to(torch.cuda.current_device())
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
    num_params = sum(p.numel() for p in gpt2.parameters() if p.requires_grad)

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(input_indices, vocab_size - 1)

    optimizer = torch.optim.SGD(gpt2.parameters(), lr=0.01)

    profiler = Profiler(os.path.join('benchmark', args.exp_name))

    num_epochs = 10
    tot_time = 0
    nproc = torch.distributed.get_world_size()

    for epoch in range(num_epochs):
        overall_name = f'gpt2_np-{nproc}_hs-{hidden_size}_nah-{num_attention_heads}_bsz-{batch_size}_num-params-{num_params}'
        profiler.start(overall_name)
        
        fname = f'gpt2_forward_np-{nproc}_hs-{hidden_size}_nl-{num_layers}_nah-{num_attention_heads}_bsz-{batch_size}_num-params-{num_params}'
        # Forward pass
        profiler.start(fname)
        loss = gpt2.forward(input_indices, position_indices, attention_mask, labels)
        train_loss = torch.mean(loss)
        # print(train_loss)
        torch.cuda.synchronize()
        profiler.stop(fname)
        # Backward pass
        bname = f'gpt2_backward_np-{nproc}_hs-{hidden_size}_nl-{num_layers}_nah-{num_attention_heads}_bsz-{batch_size}_num-params-{num_params}'
        profiler.start(bname)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        profiler.stop(bname)

        profiler.stop(overall_name)
    

if __name__ == '__main__':
    # Parse command line arguments
    parse_args()

    args = get_args()

    train()