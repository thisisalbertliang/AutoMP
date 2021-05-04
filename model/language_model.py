import torch
# from utils import get_args
from model.embedding import Embedding
from model.transformer import ParallelTransformer
from mappings import CopyToModelParallelRegion, GatherFromModelParallelRegion

def parallel_lm_logits(input_, word_embeddings_weight, parallel_output,
                       bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = CopyToModelParallelRegion.apply(input_)
    # Matrix multiply.
    if bias is None:
        logits_parallel = torch.nn.functional.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = torch.nn.functional.linear(input_parallel, word_embeddings_weight, bias)
    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return GatherFromModelParallelRegion.apply(logits_parallel)

class TransformerLanguageModel(torch.nn.Module):
    """Transformer language model.
    Arguments:
        transformer_hparams: transformer hyperparameters
        attention_mask_func: a function that takes `unmaksed-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
          masked-attention-scores = attention_mask_func(
                                     unmaksed-attention-scores, attention-mask)
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 hidden_size, vocab_size, sequence_length, hidden_dropout,
                 attention_mask_func, num_layers, layernorm_epsilon, 
                 num_attention_heads, attention_dropout,
                 init_method):
        super(TransformerLanguageModel, self).__init__()

        # self.hidden_size = hidden_size
        # self.vocab_size = vocab_size
        # self.sequence_length = sequence_length
        # self.hidden_dropout = hidden_dropout
        # self.init_method = init_method
        
        # self.num_layers = num_layers
        # self.layernorm_epsilon = layernorm_epsilon
        # self.num_attention_heads = num_attention_heads
        # self.attention_dropout = attention_dropout

        # Embeddings
        self.embedding = Embedding(hidden_size,
                                   vocab_size,
                                   sequence_length,
                                   hidden_dropout,
                                   init_method)
        self._embedding_key = 'embedding'

        # Transformer
        self.transformer = ParallelTransformer(
            attention_mask_func,
            num_layers,
            hidden_size,
            layernorm_epsilon,
            num_attention_heads, 
            attention_dropout, 
            hidden_dropout)
        self._transformer_key = 'transformer'


    def forward(self, input_ids, position_ids, attention_mask):

        # Embeddings.
        embedding_output = self.embedding(input_ids, position_ids)

        # Transformer.
        transformer_output = self.transformer(embedding_output,
                                              attention_mask)

        return transformer_output

   