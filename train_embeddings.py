import torch
import torch.distributed
from initialize import init_distributed
from arguments import parse_args, get_args
from utils import print_rank_0
from model.embedding import Embedding
from profiler import Profiler
import os


def train():
    
    # Initialize torch.distributed
    init_distributed()

    print_rank_0('AutoMP: training GPT2...')
    # Use fake train data
    batch_size = args.batch_size 
    sequence_length = args.sequence_length
    hidden_size = args.hidden_size
    vocab_size = args.vocab_size
    dropout_prob = args.hidden_dropout

    input_indices = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
    input_indices = input_indices.to(torch.cuda.current_device())
    position_indices = torch.tile(torch.arange(start=0, end=sequence_length), (batch_size, 1))
    position_indices = position_indices.to(torch.cuda.current_device())
    print_rank_0(f'AutoMP: input_indices shape = {input_indices.size()}')
    print_rank_0(f'AutoMP: position_indices shape = {position_indices.size()}')

    def init_method_normal(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
    
    embedding = Embedding(hidden_size=hidden_size, 
                          vocab_size=vocab_size, 
                          max_sequence_length=sequence_length, 
                          embedding_dropout_prob=dropout_prob, 
                          init_method=init_method_normal)

    optimizer = torch.optim.SGD(embedding.parameters(), lr=0.01)

    profiler = Profiler(os.path.join('benchmark', args.exp_name))
    

    num_epochs = 10
    tot_time = 0
    for epoch in range(num_epochs):
        overall_name = f'emb_hs-{hidden_size}'
        profiler.start(overall_name)
        
        # Forward pass
        profiler.start(f'emb_forward_hs-{hidden_size}')
        embedding_output = embedding.forward(input_indices, position_indices)
        train_loss = torch.mean(embedding_output)
        torch.cuda.synchronize()
        profiler.stop(f'emb_forward_hs-{hidden_size}')

        # Backward pass
        profiler.start(f'emb_backward_hs-{hidden_size}')
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        profiler.stop(f'emb_backward_hs-{hidden_size}')

        profiler.stop(overall_name)
        # if epoch % 50 == 0:
        # print_rank_0(f'Epoch Number {epoch}: train loss: {train_loss}, time: {time.time() - start_time}')
        # tot_time += time.time() - start_time
    # print_rank_0(f'!!! AVG EPOCH TIME: {tot_time / num_epochs}')

if __name__ == '__main__':
    # Parse command line arguments
    parse_args()

    args = get_args()

    train()