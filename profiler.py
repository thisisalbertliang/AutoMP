import time
import os
# import sys
import torch

class Profiler():
    def __init__(self, logdir):
        if torch.distributed.get_rank() == 0:
            self.time = None
            self.logdir = logdir
            # if os.path.exists(logdir):
            #     os.system(f'rm -rf {logdir}')
            # os.makedirs(logdir)

            if not os.path.exists(logdir):
                os.makedirs(logdir)
            
            self.name2startTime = {}
    
    def start(self, name):
        if torch.distributed.get_rank() == 0:
            start_time = time.time()
            self.name2startTime[name] = start_time

    def stop(self, name):
        if torch.distributed.get_rank() == 0:
            assert name in self.name2startTime

            elasped_time = time.time() - self.name2startTime[name]
            
            with open(os.path.join(self.logdir, name), 'a') as f:
                f.write(f'{elasped_time}\n')
            
            del self.name2startTime[name]
    

# def get_profiler():
#     global PROFILER
#     return PROFILER

# PROFILER = Profiler('benchmark')