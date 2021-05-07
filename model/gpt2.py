import torch
# from arguments import get_args
from model.language_model import TransformerLanguageModel, parallel_lm_logits
from model.cross_entropy import parallel_cross_entropy

class GPT2(torch.nn.Module):
    """GPT-2 Language model."""

    def __init__(self, 
                 hidden_size, vocab_size, sequence_length, hidden_dropout,
                 attention_mask_func, num_layers, layernorm_epsilon, 
                 num_attention_heads, attention_dropout,
                 init_method,
                 parallel_output=True):
        super(GPT2, self).__init__()

        self.parallel_output = parallel_output

        self.language_model = TransformerLanguageModel(
            hidden_size, vocab_size, sequence_length, hidden_dropout,
            attention_mask_func, num_layers, layernorm_epsilon, 
            num_attention_heads, attention_dropout,
            init_method
        )

    '''
    The \pyinline{forward} function of \pyinline{GPT2} model takes 4 arguments. 
    \pyinline{input_ids} is a tensor of shape \pyinline{(batch_size, sequence_length)}, 
    where each element is an index/id corresponding to a particular word. 
    \pyinline{position_ids} is a tensor of shape \pyinline{(batch_size, sequence_length)}, 
    where each element is an index/id corresponding to a paricular position in the sentence, 
    usually aranged from \pyinline{0} to \pyinline{sequence_length}. 
    \pyinline{attention_mask} is the attention mask for the attention scores of 
    the multi-headed self-attention operation.
    '''
    def forward(self, input_ids, position_ids, attention_mask, labels=None):

        # Language model.
        lm_output = self.language_model(input_ids,
                                        position_ids,
                                        attention_mask)


        # Output.
        output = parallel_lm_logits(
            lm_output,
            self.language_model.embedding.word_embeddings.weight,
            self.parallel_output)

        if labels is None:
            return output
        else:
            # print(f'Tianyu: output shape: {output.float().shape}, labels shape: {labels.shape}')
            loss = parallel_cross_entropy(output.float(), labels)
            return loss
