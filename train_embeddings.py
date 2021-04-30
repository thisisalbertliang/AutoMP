import torch
import torch.distributed
from initialize import init_distributed
from arguments import parse_args, get_args
from utils import print_rank_0
from model.embedding import Embedding


def train():
    
    # Initialize torch.distributed
    init_distributed()

    print_rank_0('AutoMP: training GPT2...')
    # Use fake train data
    batch_size = 8
    sequence_length = 1024
    hidden_size = 2048
    vocab_size = 4096
    dropout_prob = 0.1

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

    embedding_output = embedding.forward(input_indices, position_indices)
    print_rank_0(f'AutoMP: embedding_output = {embedding_output}')


if __name__ == '__main__':
    # Parse command line arguments
    parse_args()

    args = get_args()

    train()