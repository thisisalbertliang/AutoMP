# A distributed DL library for automatic tensor slicing style model parallelism on single-machine multi-gpu

## Runbook

### Example
python -m torch.distributed.launch --nproc_per_node=2 train_attention.py