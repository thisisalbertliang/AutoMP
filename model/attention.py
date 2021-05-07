import torch
import math
from utils import divide
from model.linear import ColumnParallelLinear, RowParallelLinear
from model.fused_softmax import ScaleMaskSoftmax
from mappings import split_tensor_along_last_dim

class ParallelSelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.
    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    '''
    The arguments are quite self-explanatory, with `attention_mask_func` 
    being the function that is applied before softmax to the masked entries, 
    `hidden_size` being the length of the vector of the hidden output, 
    and `attention_dropout` being the dropout rate on the attention scores.
    '''
    def __init__(self, attention_mask_func,
                 hidden_size, 
                 num_attention_heads, 
                 attention_dropout):
        super(ParallelSelfAttention, self).__init__()

        self.attention_mask_func = attention_mask_func

        # Per attention head and per partition values.
        world_size = torch.distributed.get_world_size()
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)

        # Strided linear layer.
        self.query_key_value = ColumnParallelLinear( # column linear
            hidden_size,
            3 * hidden_size,
            gather_output=False
        )

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

        self.scale_mask_softmax = ScaleMaskSoftmax(
            mask_func=self.attention_mask_func, 
            scale=None
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

        # Output.
        self.dense = RowParallelLinear(
            input_size=hidden_size, 
            output_size=hidden_size, 
            input_is_parallel=True
        )

    '''
    The forward function of this layer takes in the above two arguments, 
    with hidden_states usually being the output of the previous layer in the network, 
    and attention_mask being the attention mask used in the decoder in architectures such as GPT-2
    '''
    def forward(self, hidden_states, attention_mask):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
             3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer,
         key_layer,
         value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)


        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        
        # [b, np, sq, sk]
        output_size = (query_layer.size(1), 
                       query_layer.size(2), 
                       query_layer.size(0), 
                       key_layer.size(0))
        
        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0]*output_size[1], 
            output_size[2], 
            output_size[3],
            dtype=query_layer.dtype, 
            device=torch.cuda.current_device())

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(matmul_result, 
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0,1).transpose(1, 2),  #[b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)


        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)


        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), 
                       value_layer.size(2), 
                       query_layer.size(0), 
                       value_layer.size(3)) 

        # change view [sk, b * np, hn] 
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)
        
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)
        
        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0,1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)


        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

        return output
