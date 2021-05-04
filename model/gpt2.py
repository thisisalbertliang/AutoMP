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
            loss = parallel_cross_entropy(output.float(), labels)
            return loss
