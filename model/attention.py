import torch
import math
from utils import divide
from model.linear import ParallelLinear, RowParallelLinear
from model.fused_softmax import FusedScaleMaskSoftmax
from mappings import split_tensor_along_last_dim

class ParallelSelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.
    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    # def __init__(self, attention_mask_func, init_method,
    #              output_layer_init_method, layer_number, 
    #              hidden_size, num_attention_heads, attention_dropout):
    def __init__(self, layer_number, hidden_size, num_attention_heads, attention_dropout, attention_mask_func=None):
        super(ParallelSelfAttention, self).__init__()
        # args = get_args()
        # self.fp16 = args.fp16

        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = True #args.apply_query_key_layer_scaling
        # self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        # Per attention head and per partition values.
        world_size = torch.distributed.get_world_size()
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)

        # Strided linear layer.
        self.query_key_value = ParallelLinear( # column linear
            hidden_size,
            3 * hidden_size,
            gather=False)
            # gather_output=False,
            # init_method=init_method)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            False, # self.fp16,
            False, # args.scaled_upper_triang_masked_softmax_fusion,
            False, # args.scaled_masked_softmax_fusion,
            self.attention_mask_func,
            True, #self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

        # Output.
        self.dense = RowParallelLinear(
            hidden_size,
            hidden_size,
            input_is_parallel=True,
            # init_method=output_layer_init_method,
            skip_bias_add=True)

    # def _transpose_last_dim(self, mixed_layer, num_splits, num_splits_first):
    #     input_shape = mixed_layer.size();
    #     if num_splits_first:
    #         """[s, b, num_splits * np * hn] 
    #         -->(view) [s, b, num_splits, np, hn] 
    #         -->(tranpose) [s, b, np, num_splits, hn] 
    #         -->(view) [s, b, np * num_splits * hn] """

    #         intermediate_shape = input_shape[:-1] +\
    #             (num_splits, self.num_attention_heads_per_partition,
    #              self.hidden_size_per_attention_head)

    #         mixed_layer = mixed_layer.view(*intermediate_shape)
    #         mixed_layer = mixed_layer.transpose(-2, -3).contiguous()
    #     else:
    #         """[s, b, np * hn * num_splits] 
    #         -->(view) [s, b, np, hn, num_splits] 
    #         -->(tranpose) [s, b, np, num_splits, hn] 
    #         -->(view) [s, b, np * num_splits * hn] """

    #         intermediate_shape = input_shape[:-1] +\
    #             (self.num_attention_heads_per_partition,
    #              self.hidden_size_per_attention_head, num_splits)

    #         mixed_layer = mixed_layer.view(*intermediate_shape)
    #         mixed_layer = mixed_layer.transpose(-1, -2).contiguous()
    #     mixed_layer = mixed_layer.view(*input_shape)
        
    #     return mixed_layer

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        # mixed_x_layer, _ = self.query_key_value(hidden_states)
        mixed_x_layer = self.query_key_value(hidden_states)

        # checkpoint_version = get_checkpoint_version()
        # if checkpoint_version is not None:
        #    if checkpoint_version == 0:
        #        # [s, b, (3 * np * hn)] --> [s, b, (np * 3 * hn)]
        #        mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 3, True)
        #    elif checkpoint_version == 1.0:
        #        # [s, b, (np * hn * 3)] --> [s, b, (np * 3 * hn)]
        #        mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 3, False)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
             3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer,
         key_layer,
         value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)


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


        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                        ...,
                        attention_scores.size(3) - 1,
                        :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                        ...,
                        :attention_scores.size(3),
                        :attention_scores.size(3)]


        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        # with mpu.get_cuda_rng_tracker().fork():
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

        output, bias = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output, bias
