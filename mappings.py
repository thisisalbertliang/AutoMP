import torch
import torch.distributed
from utils import divide


def split_tensor_along_last_dim(tensor, num_partitions,
                                contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous in memory.
    """
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)

    # Split
    tensor_list = torch.split(tensor,
                              split_size_or_sections=last_dim_size,
                              dim=last_dim)

    return tensor_list


class CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        torch.distributed.all_reduce(grad_output)


class GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_):
        """Gather tensors and concatinate along the last dimension."""
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        last_dim = input_.dim() - 1

        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        tensor_list[rank] = input_
        torch.distributed.all_gather(tensor_list, input_)

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=last_dim).contiguous()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Split the tensor along its last dimension and keep the corresponding slice."""
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        grad_partition_list = split_tensor_along_last_dim(grad_output, world_size)

        # Note: torch.split does not create contiguous tensors by default.
        grad_partition = grad_partition_list[rank].contiguous()
        return grad_partition

class ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        """All-reduce the input tensor across model parallel group."""
        # Bypass the function if we are using only 1 GPU
        if torch.distributed.get_world_size() == 1:
            return input_
        
        # All-reduce
        torch.distributed.all_reduce(input_)

        return input_
    
    @staticmethod
    def backward(ctx, grad_output):
        # identity function
        return grad_output

def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = torch.distributed.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = torch.distributed.get_rank()
    output = input_list[rank].contiguous()

    return output


def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = torch.distributed.get_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output

class ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)
