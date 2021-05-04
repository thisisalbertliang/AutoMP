import torch
# from arguments import get_args
from model.attention import ParallelSelfAttention
from model.linear import ColumnParallelLinear, RowParallelLinear

class ParallelMLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """
    def __init__(self, hidden_size):
        super(ParallelMLP, self).__init__()

        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(
            hidden_size,
            4 * hidden_size,
            gather_output=False
        )

        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            4 * hidden_size,
            hidden_size,
            input_is_parallel=True
        )
    
    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)

        intermediate_parallel = torch.nn.functional.gelu(
            intermediate_parallel
        )

        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class ParallelTransformerLayer(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """
    def __init__(self, attention_mask_func, layer_number, 
                 hidden_size, layernorm_epsilon, 
                 num_attention_heads, attention_dropout, 
                 hidden_dropout):
        super(ParallelTransformerLayer, self).__init__()

        self.layer_number = layer_number

        # Layernorm on the input data
        self.input_layernorm = torch.nn.LayerNorm(
            hidden_size, eps=layernorm_epsilon
        ).cuda()

        # Self attention
        self.attention = ParallelSelfAttention(
            attention_mask_func=attention_mask_func, 
            hidden_size=hidden_size, 
            num_attention_heads=num_attention_heads, 
            attention_dropout=attention_dropout
        )

        self.hidden_dropout = hidden_dropout
    
        # Layernorm on the input data.
        self.post_attention_layernorm = torch.nn.LayerNorm(
            hidden_size, eps=layernorm_epsilon
        ).cuda()

        # MLP
        self.mlp = ParallelMLP(hidden_size=hidden_size)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [b, s, h]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output = self.attention(
            layernorm_output,
            attention_mask
        )
    
        layernorm_input = torch.nn.functional.dropout(
            attention_output, 
            p=self.hidden_dropout
        )

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(
            layernorm_input
        )

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        output = torch.nn.functional.dropout(
            mlp_output, 
            p=self.hidden_dropout
        )

        return output


class ParallelTransformer(torch.nn.Module):

    def __init__(self, 
                 attention_mask_func,
                 num_layers, 
                 hidden_size, 
                 layernorm_epsilon, 
                 num_attention_heads, 
                 attention_dropout, 
                 hidden_dropout):

        super(ParallelTransformer, self).__init__()

        self.num_layers = num_layers
        
        # Transformer Layers
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                attention_mask_func=attention_mask_func, 
                layer_number=layer_number, 
                hidden_size=hidden_size,
                layernorm_epsilon=layernorm_epsilon, 
                num_attention_heads=num_attention_heads, 
                attention_dropout=attention_dropout, 
                hidden_dropout=hidden_dropout
            )
        self.layers = torch.nn.ModuleList(
                [build_layer(i + 1) for i in range(self.num_layers)])
        
        # Final layer norm before output.
        self.final_layernorm = torch.nn.LayerNorm(
            hidden_size,
            eps=layernorm_epsilon
        ).cuda()

    def forward(self, hidden_states, attention_mask):
        # data format change to avoid explicit tranposes : [b s h] --> [s b h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        for index in range(self.num_layers):
            layer = self.layers[index]
            hidden_states = layer(hidden_states, attention_mask)
        
        # reverting data format change [s b h] --> [b s h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()
    
        # Final layer norm.
        output = self.final_layernorm(hidden_states)

        return output
