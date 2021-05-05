import torch
import torch.distributed
import numpy as np
import time
from initialize import init_distributed
from arguments import parse_args, get_args
from utils import print_rank_0, get_ltor_masks_and_position_ids
from model.transformer import ParallelTransformerLayer
from profiler import Profiler
import os
from model.embedding import Embedding
from utils import divide


def train():
    
    # Initialize torch.distributed
    init_distributed()

    print_rank_0('AutoMP: training ParallelTransformerLayer...')

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

    def init_method_normal(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
    
    embedding = Embedding(hidden_size=hidden_size, 
                          vocab_size=vocab_size, 
                          max_sequence_length=sequence_length, 
                          embedding_dropout_prob=hidden_dropout, 
                          init_method=init_method_normal)
    embedding_output = embedding.forward(input_indices, position_indices)

    transformer_layer = ParallelTransformerLayer(
        attention_mask_func=gpt2_attention_mask_func, 
        layer_number=0, 
        hidden_size=hidden_size, 
        layernorm_epsilon=layernorm_epsilon, 
        num_attention_heads=num_attention_heads, 
        attention_dropout=attention_dropout, 
        hidden_dropout=hidden_dropout
    )

    # attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(input_indices, vocab_size - 1)
    attention_mask = (torch.randint(low=0, high=2, 
                                    size=(sequence_length, num_attention_heads, batch_size, batch_size))
                      < 0).cuda()

    optimizer = torch.optim.SGD(transformer_layer.parameters(), lr=0.01)

    profiler = Profiler(os.path.join('benchmark', args.exp_name))

    num_epochs = 10
    tot_time = 0
    for epoch in range(num_epochs):
        input_ = torch.rand(size=embedding_output.size()).cuda()

        overall_name = f'transformer_layer_hs-{hidden_size}_nah-{num_attention_heads}_bsz-{batch_size}'
        profiler.start(overall_name)

        fname = f'transformer_layer_forward_hs-{hidden_size}_nah-{num_attention_heads}_bsz-{batch_size}'
        # Forward pass
        profiler.start(fname)
        loss = transformer_layer.forward(input_, attention_mask)
        train_loss = torch.mean(loss)
        # print(train_loss)
        torch.cuda.synchronize()
        profiler.stop(fname)
        # Backward pass
        bname = f'transformer_layer_backward_hs-{hidden_size}_nah-{num_attention_heads}_bsz-{batch_size}'
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