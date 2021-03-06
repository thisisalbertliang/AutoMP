import torch
import torch.distributed
from initialize import init_distributed
from arguments import parse_args, get_args
from utils import print_rank_0, get_ltor_masks_and_position_ids
from model.embedding import Embedding
from model.attention import ParallelSelfAttention

# def bert_attention_mask_func(attention_scores, attention_mask):
#     attention_scores.masked_fill_(attention_mask, -10000.0)
#     return attention_scores

def train():
    
    # Initialize torch.distributed
    init_distributed()

    print_rank_0('AutoMP: training self attention layer...')
    # Use fake train data
    args = get_args()
    batch_size = 32
    sequence_length = 1024
    hidden_size = args.hidden_size
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
    # print_rank_0(f'AutoMP: embedding_output = {embedding_output}')

    def gpt2_attention_mask_func(attention_scores, ltor_mask):

        print(f'ALBERT_DEBUG: attention_scores.size() = {attention_scores.size()}')
        print(f'ALBERT_DEBUG: ltor_mask.size() = {ltor_mask.size()}')

        attention_scores.masked_fill_(ltor_mask, -10000.0)
        return attention_scores

    self_attention = ParallelSelfAttention(
        attention_mask_func=gpt2_attention_mask_func, 
        hidden_size=args.hidden_size, 
        num_attention_heads=args.num_attention_heads, 
        attention_dropout=0.1
    )

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(input_indices, vocab_size - 1)

    print(f'ALBERT_DEBUG: embedding_output.size() = {embedding_output.size()}')

    self_att_output = self_attention.forward(hidden_states=embedding_output, attention_mask=attention_mask)
    print_rank_0(f'AutoMP: self_att_output = {self_att_output}')



if __name__ == '__main__':
    # Parse command line arguments
    parse_args()

    args = get_args()

    train()