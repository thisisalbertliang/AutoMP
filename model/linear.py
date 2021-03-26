import torch
import torch.distributed
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from mappings import CopyToModelParallelRegion, GatherFromModelParallelRegion


class ParallelLinear(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, gather: bool = True):
        super(ParallelLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gather = gather

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
        torch.nn.init.xavier_normal(self.weight)

        self.bias = Parameter(torch.empty(
            self.output_size_per_partition,
            device=torch.cuda.current_device(),
            dtype=torch.float
        ))
        # Initialize bias to zero
        with torch.no_grad():
            self.bias.zero_()

    def forward(self, input_):
        # Set up backprop all-reduce
        # print('HERE00', input_.shape)
        input_parallel = CopyToModelParallelRegion.apply(input_)
        # print('HERE11', input_parallel.shape)

        # Matrix multiply
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        output_parallel = F.relu(output_parallel)

        if not self.gather:
            return output_parallel

        # print('HERE0', output_parallel.shape)
        output_gathered = GatherFromModelParallelRegion.apply(output_parallel)
        # print('HERE1', output_gathered.shape)

        return output_gathered

