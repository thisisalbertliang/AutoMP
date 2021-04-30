import torch
import torch.distributed
from utils import vocab_range_from_per_partition_vocab_size
from mappings import ReduceFromModelParallelRegion

class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """
    def __init__(self, num_embeddings, embedding_dim, init_method=torch.nn.init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
            
        self.model_parallel_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()

        # Divide the weight matrix along the vocabulary dimension
        self.vocab_start_index, self.vocab_end_index = \
            vocab_range_from_per_partition_vocab_size(
                self.num_embeddings, self.rank
            )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        # Allocate weights and initialize

        print(f'ALBERT_DEBUG: torch.cuda.current_device() = {torch.cuda.current_device()}')

        self.weight = torch.nn.Parameter(torch.empty(
            self.num_embeddings_per_partition, self.embedding_dim, 
            device=torch.cuda.current_device()
        ))
        init_method(self.weight)

    def forward(self, input_: torch.Tensor):
        # input shape: (batch_size, sequence_length)
        # input elements are vocab indices

        print(f'ALBERT_DEBUG: input_.device = {input_.device}')

        if self.model_parallel_size > 1:
            # Build the mask
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
        
        print(f'ALBERT_DEBUG: masked_input.device = {masked_input.device}')
        print(f'ALBERT_DEBUG: self.weight.device = {self.weight.device}')

        # Get the embeddings
        output_parallel = torch.nn.functional.embedding(masked_input, self.weight)

        # Mask the output embedding
        if self.model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs
        output = ReduceFromModelParallelRegion.apply(output_parallel)
        return output


class Embedding(torch.nn.Module):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """
    def __init__(self, 
                 hidden_size, 
                 vocab_size, 
                 max_sequence_length, 
                 embedding_dropout_prob, 
                 init_method):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        
        # Word embedding (vocab parallel)
        self.word_embeddings = VocabParallelEmbedding(
            vocab_size, self.hidden_size, init_method=init_method
        )
        
        # Positional embedding (serial)
        self.position_embeddings = torch.nn.Embedding(
            num_embeddings=max_sequence_length, embedding_dim=self.hidden_size
        ).cuda(torch.cuda.current_device())
        # Initialize the positional embeddings
        self.init_method(self.position_embeddings.weight)

        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
    
    def forward(self, input_indices, position_indices):

        print(f'ALBERT_DEBUG: input_indices.device = {input_indices.device}')
        print(f'ALBERT_DEBUG: position_indices.device = {position_indices.device}')

        # Embeddings
        words_embeddings = self.word_embeddings.forward(input_indices)
        position_embeddings = self.position_embeddings.forward(position_indices)
        embeddings = words_embeddings + position_embeddings

        # Dropout
        embeddings = self.embedding_dropout(embeddings)

        return embeddings

