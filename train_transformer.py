import torch
import torch.distributed
from initialize import init_distributed
from arguments import parse_args, get_args
from utils import print_rank_0, get_ltor_masks_and_position_ids
from model.embedding import Embedding
from model.transformer import ParallelTransformerLayer, ParallelTransformer
from model.attention import ParallelSelfAttention

def train():
    
    # Initialize torch.distributed
    init_distributed()

    print_rank_0('AutoMP: training GPT2...')
    # Use fake train data
    args = get_args()
    sequence_length = 1024
    vocab_size = 4096
    dropout_prob = 0.1

    input_indices = torch.randint(low=0, high=vocab_size, size=(args.batch_size, sequence_length))
    input_indices = input_indices.to(torch.cuda.current_device())
    position_indices = torch.tile(torch.arange(start=0, end=sequence_length), (args.batch_size, 1))
    position_indices = position_indices.to(torch.cuda.current_device())
    print_rank_0(f'AutoMP: input_indices shape = {input_indices.size()}')
    print_rank_0(f'AutoMP: position_indices shape = {position_indices.size()}')

    def init_method_normal(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
    embedding = Embedding(hidden_size=args.hidden_size, 
              vocab_size=vocab_size, 
              max_sequence_length=sequence_length, 
              embedding_dropout_prob=dropout_prob, 
              init_method=init_method_normal)

    embedding_output = embedding.forward(input_indices, position_indices)
    # print_rank_0(f'AutoMP: embedding_output = {embedding_output}')

    def gpt2_attention_mask_func(attention_scores, ltor_mask):
        attention_scores.masked_fill_(ltor_mask, -10000.0)
        return attention_scores

    transformer = ParallelTransformer(
        attention_mask_func=gpt2_attention_mask_func, 
        num_layers=args.num_layers, 
        hidden_size=args.hidden_size, 
        layernorm_epsilon=args.layernorm_epsilon, 
        num_attention_heads=args.num_attention_heads,
        attention_dropout=0.1, 
        hidden_dropout=0.1
    )

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(input_indices, vocab_size - 1)

    transformer_output = transformer.forward(hidden_states=embedding_output, attention_mask=attention_mask)
    print_rank_0(f'AutoMP: transformer_output = {transformer_output}')



if __name__ == '__main__':
    # Parse command line arguments
    parse_args()

    args = get_args()

    train()