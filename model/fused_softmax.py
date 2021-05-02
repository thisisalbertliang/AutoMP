import torch

class ScaleMaskSoftmax(torch.nn.Module):
    """
       operation: scaling + mask + softmax
       Arguments:
           upper_triang_mask: if true, apply upper triangular masking.
                              (used in gpt family networks)
           mask_func: mask function to be applied.
           scale: scaling factor used in input tensor scaling.
    """
    def __init__(self, mask_func, scale):
        super(ScaleMaskSoftmax, self).__init__()
        self.mask_func = mask_func
        self.scale = scale

    def forward(self, input_, mask):
        # [b, np, s, s]
        assert input_.dim() == 4 

        mask_output = self.mask_func(input_, mask)           
        if self.scale is not None:
            mask_output = mask_output * self.scale             
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        return probs
