import torch
import torch.distributed
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from mappings import CopyToModelParallelRegion, GatherFromModelParallelRegion, ReduceFromModelParallelRegion, ScatterToModelParallelRegion
from utils import divide


class ColumnParallelLinear(torch.nn.Module):
    '''
    `input_size` is the size of the feature dimension of each input sample. 
    `output_size` is the size of the feature dimension of each output sample. 
    `gather` is set to `True`, then `ColumnParallelLinear` will perform an all-gather operation 
     at the end of its `forward` function to ensure that every GPU has a copy of the full output activation. 
     If `gather` is set to `False`, then at the end of the `forward` function, every GPU only retains its own partition of the output activation. 
    '''
    def __init__(self, input_size: int, output_size: int, gather_output: bool = True):
        super(ColumnParallelLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output

        world_size = torch.distributed.get_world_size()
        assert output_size % world_size == 0, \
            'AutoMP requires the output dimension of linear map to be divisible by world size'
        self.output_size_per_partition = output_size // world_size

        # torch.nn.functional.linear performs XA^T + b
        # so we allocate the transpose
        self.weight = Parameter(torch.empty(
            self.output_size_per_partition, self.input_size,
            device=torch.cuda.current_device(),
            dtype=torch.float
        ))
        torch.nn.init.xavier_normal_(self.weight)

        self.bias = Parameter(torch.empty(
            self.output_size_per_partition,
            device=torch.cuda.current_device(),
            dtype=torch.float
        ))
        # Initialize bias to zero
        with torch.no_grad():
            self.bias.zero_()

    def forward(self, input_):

        # print(f'ALBERT_DEBUG: input_.size() = {input_.size()}')
        # print(f'ALBERT_DEBUG: self.weight.size() = {self.weight.size()}')

        # Set up backprop all-reduce
        input_parallel = CopyToModelParallelRegion.apply(input_)

        # Matrix multiply
        # print(f'ALBERT_DEBUG: input_parallel.size() = {input_parallel.size()}')
        # print(f'ALBERT_DEBUG: self.weight.size() = {self.weight.size()}')
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        output_parallel = F.relu(output_parallel)

        if not self.gather_output:
            return output_parallel
        
        output_gathered = GatherFromModelParallelRegion.apply(output_parallel)

        return output_gathered


class RowParallelLinear(torch.nn.Module):
    """
    The first two are semantically identical to those of the column parallel version. 
    The third argument indicates whether the input to this layer during a forward pass is parallelized already. 
    If not, AutoMP will split the input and keep only the corresponding chuck to the current GPU.
    """

    def __init__(self, 
                 input_size, 
                 output_size,
                 input_is_parallel=False):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = torch.distributed.get_world_size()
        self.input_size_per_partition = divide(input_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        self.weight = Parameter(torch.empty(
            self.output_size, self.input_size_per_partition,
            device=torch.cuda.current_device(), dtype=torch.float))
        torch.nn.init.xavier_normal_(self.weight)

        self.bias = Parameter(torch.empty(
            self.output_size, device=torch.cuda.current_device(),
            dtype=torch.float))
        # Always initialize bias to zero.
        with torch.no_grad():
            self.bias.zero_()


    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = ScatterToModelParallelRegion.apply(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output = ReduceFromModelParallelRegion.apply(output_parallel)

        output = output + self.bias

        return output