import matplotlib.pyplot as plt
from enum import Enum
from statistics import mean
import numpy as np
import pandas as pd
from collections import OrderedDict

VOCAB_SIZES = [1024, 3072, 8192, 32768, 65536, 131072, 262144, 524288]
NUM_GPUS = [1, 2, 4, 8]
HIDDEN_SIZES = [256, 512, 1024, 1536, 1920, 2304, 3072, 4096]
NUM_ATTENTION_HEADS = [8, 16, 32, 64]
NP_HS_NHEADS_2_NPARAM = {(8, 256, 8): 1087136, (8, 256, 16): 1087136, (8, 256, 32): 1087136, (8, 256, 64): 1087136, (4, 256, 8): 1382720, (4, 256, 16): 1382720, (4, 256, 32): 1382720, (4, 256, 64): 1382720, (2, 256, 8): 1973888, (2, 256, 16): 1973888, (2, 256, 32): 1973888, (2, 256, 64): 1973888, (8, 512, 8): 2764096, (8, 512, 16): 2764096, (8, 512, 32): 2764096, (8, 512, 64): 2764096, (1, 256, 8): 3156224, (1, 256, 16): 3156224, (1, 256, 32): 3156224, (4, 512, 8): 3945088, (4, 512, 16): 3945088, (4, 512, 32): 3945088, (4, 512, 64): 3945088, (2, 512, 8): 6307072, (2, 512, 16): 6307072, (2, 512, 32): 6307072, (2, 512, 64): 6307072, (8, 1024, 8): 7887488, (8, 1024, 16): 7887488, (8, 1024, 32): 7887488, (8, 1024, 64): 7887488, (1, 512, 8): 11031040, (1, 512, 16): 11031040, (1, 512, 32): 11031040, (4, 1024, 8): 12608768, (4, 1024, 16): 12608768, (4, 1024, 32): 12608768, (4, 1024, 64): 12608768, (8, 1536, 8): 15370176, (8, 1536, 16): 15370176, (8, 1536, 32): 15370176, (8, 1536, 64): 15370176, (2, 1024, 8): 22051328, (2, 1024, 16): 22051328, (2, 1024, 32): 22051328, (2, 1024, 64): 22051328, (8, 1920, 8): 22530480, (8, 1920, 16): 22530480, (8, 1920, 32): 22530480, (8, 1920, 64): 22530480, (4, 1536, 8): 25991040, (4, 1536, 16): 25991040, (4, 1536, 32): 25991040, (4, 1536, 64): 25991040, (8, 2304, 8): 31017888, (8, 2304, 16): 31017888, (8, 2304, 32): 31017888, (8, 2304, 64): 31017888, (4, 1920, 8): 39124320, (4, 1920, 16): 39124320, (4, 1920, 32): 39124320, (4, 1920, 64): 39124320, (1, 1024, 8): 40936448, (1, 1024, 16): 40936448, (1, 1024, 32): 40936448, (2, 1536, 8): 47232768, (2, 1536, 16): 47232768, (2, 1536, 32): 47232768, (2, 1536, 64): 47232768, (8, 3072, 8): 51974016, (8, 3072, 16): 51974016, (8, 3072, 32): 51974016, (8, 3072, 64): 51974016, (4, 2304, 8): 54911808, (4, 2304, 16): 54911808, (4, 2304, 32): 54911808, (4, 2304, 64): 54911808, (2, 1920, 8): 72312000, (2, 1920, 16): 72312000, (2, 1920, 32): 72312000, (2, 1920, 64): 72312000, (8, 4096, 8): 88173056, (8, 4096, 16): 88173056, (8, 4096, 32): 88173056, (8, 4096, 64): 88173056, (1, 1536, 8): 89716224, (1, 1536, 16): 89716224, (1, 1536, 32): 89716224, (4, 3072, 8): 94449408, (4, 3072, 16): 94449408, (4, 3072, 32): 94449408, (4, 3072, 64): 94449408, (2, 2304, 8): 102699648, (2, 2304, 16): 102699648, (2, 2304, 32): 102699648, (1, 1920, 8): 138687360, (1, 1920, 16): 138687360, (4, 4096, 8): 163681280, (4, 4096, 16): 163681280, (4, 4096, 32): 163681280, (4, 4096, 64): 163681280, (2, 3072, 8): 179400192, (2, 3072, 16): 179400192, (2, 3072, 32): 179400192, (1, 2304, 8): 198275328, (1, 2304, 16): 198275328, (2, 4096, 8): 314697728, (2, 4096, 16): 314697728, (2, 4096, 32): 314697728, (1, 3072, 8): 349301760, (1, 3072, 16): 349301760, (1, 4096, 8): 616730624}
NPARAM_2_NP_HS_NHEADS = None

class Phase(Enum):
    FORWARD = '_forward_'
    BACKWARD = '_backward_'
    TOTAL = '_'

    def __str__(self):
        if self == Phase.FORWARD:
            return '_forward_'
        elif self == Phase.BACKWARD:
            return '_backward_'
        else:
            return '_'
    
    def get_pretty_name(self):
        if self == Phase.FORWARD:
            return 'Forward Pass'
        elif self == Phase.BACKWARD:
            return 'Backward Pass'
        else:
            return ''

PHASES = [Phase.FORWARD, Phase.BACKWARD, Phase.TOTAL]

MODEL_SIZES_IN_MILLIONS = [5.8368, 78.00248, 616.91448, 831.64248, 1640.95956, 3280.95216, 4374.28056, 6560.93736]

AVG_ELAPSED_TIME_PER_EPOCH_1_GPU = [0.009878396988, 0.1116950512, 0.8001461029, 1.076288056]
AVG_ELAPSED_TIME_PER_EPOCH_2_GPU = [0.009790420532, 0.06844973564, 0.4648096561, 0.6129128456, 1.187486506]
AVG_ELAPSED_TIME_PER_EPOCH_4_GPU = [0.005018949509, 0.03479409218, 0.3083040714, 0.3655611753, 0.628198576, 1.402994871]
AVG_ELAPSED_TIME_PER_EPOCH_8_GPU = [0.004081726074, 0.02193212509, 0.1242339611, 0.2516739368, 0.3996932983, 0.7051727533, 0.7922039032, 1.498698711]

NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_0 = {1: 0.009878396988, 2: 0.009790420532, 4: 0.005018949509, 8: 0.004081726074}
NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_1 = {1: 0.1116950512, 2: 0.06844973564, 4: 0.03479409218, 8: 0.02193212509}
NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_2 = {1: 0.8001461029, 2: 0.4648096561, 4: 0.3083040714, 8: 0.1242339611}
NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_3 = {1: 1.076288056, 2: 0.6129128456, 4: 0.3655611753, 8: 0.2516739368}
NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_4 = {2: 1.187486506, 4: 0.628198576, 8: 0.3996932983}
NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_5 = {4: 1.402994871, 8: 0.7051727533}
NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_6 = {8: 0.7922039032}
NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_7 = {8: 1.498698711}


def plot_avg_elasped_time_vs_model_size(avg_elasped_time_per_epoch, num_gpus):
    plt.scatter(MODEL_SIZES_IN_MILLIONS[:len(avg_elasped_time_per_epoch)], avg_elasped_time_per_epoch)
    plt.title(f'Average Elasped Time per Epoch v.s. Model Size (Using {num_gpus} GPU)')
    plt.xlabel('Model Size (Number of Parameters in Millions)')
    plt.ylabel('Average Elasped Time per Epoch in Seconds')
    plt.legend()
    plt.show()


def plot_throughput_vs_model_size(throughputs, num_gpus):
    plt.scatter(MODEL_SIZES_IN_MILLIONS[:len(throughputs)], throughputs)
    plt.title(f'Throughput v.s. Model Size (Using {num_gpus} GPU)')
    plt.xlabel('Model Size (Number of Parameters in Millions)')
    plt.ylabel('Throughput (Number of Training Samples Processed per Second)')
    plt.legend()
    plt.show()


# def average_elasped_time_vs_model_size_1_gpu():
#     avg_elasped_time_per_epoch = [0.009878396988, 0.1116950512, 0.8001461029, 1.076288056]
#     plot_avg_elasped_time_vs_model_size(avg_elasped_time_per_epoch, 1)
#
#
# def average_elasped_time_vs_model_size_2_gpu():
#     avg_elasped_time_per_epoch = [0.009790420532, 0.06844973564, 0.4648096561, 0.6129128456, 1.187486506]
#     plot_avg_elasped_time_vs_model_size(avg_elasped_time_per_epoch, 2)
#
#
# def average_elasped_time_vs_model_size_4_gpu():
#     avg_elasped_time_per_epoch = [0.005018949509, 0.03479409218, 0.3083040714, 0.3655611753, 0.628198576, 1.402994871]
#     plot_avg_elasped_time_vs_model_size(avg_elasped_time_per_epoch, 4)
#
#
# def average_elasped_time_vs_model_size_8_gpu():
#     avg_elasped_time_per_epoch = [0.004081726074, 0.02193212509, 0.1242339611, 0.2516739368, 0.3996932983, 0.7051727533, 0.7922039032, 1.498698711]
#     plot_avg_elasped_time_vs_model_size(avg_elasped_time_per_epoch, 8)


def mlp_average_elapsed_time_vs_model_size_all():
    plt.scatter(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_1_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_1_GPU)
    plt.plot(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_1_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_1_GPU, label='1 GPU')
    plt.scatter(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_2_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_2_GPU)
    plt.plot(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_2_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_2_GPU, label='2 GPU')
    plt.scatter(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_4_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_4_GPU)
    plt.plot(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_4_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_4_GPU, label='4 GPU')
    plt.scatter(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_8_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_8_GPU)
    plt.plot(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_8_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_8_GPU, label='8 GPU')
    plt.title(f'MLP: Average Elapsed Time per Epoch v.s. Model Size')
    plt.xlabel('Model Size (Number of Parameters in Millions)')
    plt.ylabel('Average Elapsed Time per Epoch in Seconds')
    plt.legend()
    plt.savefig('plots/mlp_avg_elasped_time_vs_model_size.png')
    plt.show()


def mlp_average_elapsed_time_vs_num_gpus_all():
    plt.figure(figsize=(10,8), dpi=100)
    plt.scatter(x=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_0.keys(),
                y=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_0.values())
    plt.plot(NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_0.keys(),
                NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_0.values(),
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[0]:.1f}M')
    plt.scatter(x=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_1.keys(),
                y=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_1.values())
    plt.plot(NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_1.keys(),
                NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_1.values(),
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[1]:.1f}M')
    plt.scatter(x=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_2.keys(),
                y=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_2.values())
    plt.plot(NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_2.keys(),
                NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_2.values(),
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[2]:.1f}M')
    plt.scatter(x=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_3.keys(),
                y=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_3.values())
    plt.plot(NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_3.keys(),
                NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_3.values(),
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[3]:.1f}M')
    plt.scatter(x=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_4.keys(),
                y=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_4.values())
    plt.plot(NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_4.keys(),
                NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_4.values(),
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[4]:.1f}M')
    plt.scatter(x=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_5.keys(),
                y=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_5.values())
    plt.plot(NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_5.keys(),
                NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_5.values(),
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[5]:.1f}M')
    plt.scatter(x=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_6.keys(),
                y=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_6.values())
    plt.plot(NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_6.keys(),
                NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_6.values(),
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[6]:.1f}M')
    plt.scatter(x=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_7.keys(),
                y=NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_7.values())
    plt.plot(NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_7.keys(),
                NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_7.values(),
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[7]:.1f}M')
    plt.title(f'MLP: Average Elapsed Time per Epoch v.s. Number of GPUs')
    plt.xlabel('Number of GPUs')
    plt.ylabel('Average Elapsed Time per Epoch in Seconds')
    plt.legend()
    plt.savefig('plots/mlp_avg_elapsed_time_vs_num_gpus2.png')
    plt.show()


def get_emb_runtime(phase: Phase, nproc: int, vocab_size: int):
    with open(f'benchmark/exp/emb{phase}np-{nproc}_vs-{vocab_size}', 'r') as f:
        runtime_list = [float(l) for l in f.read().splitlines()][1:]
        mean_runtime = mean(runtime_list)
        return mean_runtime

def get_transformer_layer_runtime(phase: Phase, nproc: int, hidden_size: int, 
                                  num_attention_heads: int, batch_size: int):
    with open(f'benchmark/exp/transformer_layer{phase}np-{nproc}_hs-{hidden_size}_nah-{num_attention_heads}_bsz-{batch_size}', 'r') as f:
        runtime_list = [float(l) for l in f.read().splitlines()][1:]
        mean_runtime = mean(runtime_list)
        return mean_runtime

def get_gpt2_runtime(phase: Phase, nproc: int, hidden_size: int, 
                     num_attention_heads: int, batch_size: int):
    try:
        num_params = NP_HS_NHEADS_2_NPARAM[(nproc, hidden_size, num_attention_heads)]
        if phase == Phase.TOTAL:
            filename = f'benchmark/exp/gpt2_np-{nproc}_hs-{hidden_size}_nah-{num_attention_heads}_bsz-{batch_size}_num-params-{num_params}'
        else:
            filename = f'benchmark/exp/gpt2{phase}np-{nproc}_hs-{hidden_size}_nl-3_nah-{num_attention_heads}_bsz-{batch_size}_num-params-{num_params}'
        with open(filename, 'r') as f:
            runtime_list = [float(l) for l in f.read().splitlines()][1:]
            mean_runtime = mean(runtime_list)
            return mean_runtime
    except:
        print('get_gpt2_runtime excepted')
        return None


def plot_emb_runtime_vs_vocab_size():
    for phase in PHASES:
        for num_gpu in NUM_GPUS:
            runtimes = [get_emb_runtime(phase, num_gpu, vs) for vs in VOCAB_SIZES]
            plt.scatter(VOCAB_SIZES, runtimes, label=f'{num_gpu} GPUs')
            plt.plot(VOCAB_SIZES, runtimes)
        phase_title = 'Forward + Backward Pass' if phase == Phase.TOTAL else phase.get_pretty_name()
        plt.title(f'Embedding Layer {phase_title}:\n'
                  f'Average Elasped Time per Pass v.s. Vocab Size (Using {NUM_GPUS[-1]} GPU)')
        plt.xlabel('Vocab Size')
        plt.ylabel('Average Elasped Time per Pass in Seconds')
        plt.legend()
        plt.savefig(f'plots/emb{phase}runtime_vs_vocab_size.png')
        plt.clf()

def plot_emb_runtime_vs_num_gpus():
    for phase in PHASES:
        for vs in VOCAB_SIZES:
            runtimes = [get_emb_runtime(phase, num_gpu, vs) for num_gpu in NUM_GPUS]
            plt.scatter(NUM_GPUS, runtimes, label=f'Vocab Size {vs}')
            plt.plot(NUM_GPUS, runtimes)
        phase_title = 'Forward + Backward Pass' if phase == Phase.TOTAL else phase.get_pretty_name()
        plt.title(f'Embedding Layer {phase_title}:\n'
                  f'Average Elasped Time per Pass v.s. Number of GPUs')
        plt.xlabel('Number of GPUs')
        plt.ylabel('Average Elapsed Time per Pass in Seconds')
        plt.legend()
        plt.savefig(f'plots/emb{phase}runtime_vs_num_gpus.png')
        plt.clf()
        
# def plot_emb_runtime_vs_num_gpus_bar():
#     bottom = None
#     vs = 8192
#     width = 1
#     for phase in PHASES:
#         if phase == Phase.TOTAL:
#             continue
#         if phase == Phase.FORWARD:
#             c = 'r'
#         elif phase == Phase.BACKWARD:
#             c = 'g'    
#         runtimes = [get_emb_runtime(phase, num_gpu, vs) for num_gpu in NUM_GPUS[:1]]
#         if bottom:
#             plt.bar(NUM_GPUS, runtimes, width=width, bottom=bottom, color=c)
#             bottom += np.array(runtimes)
#         else:
#             plt.bar(NUM_GPUS, runtimes, width=width, color=c)
#             bottom = np.array(runtimes)
        
#         phase_title = phase.get_pretty_name()
#         plt.title(f'Embedding Layer {phase_title}: Average Elasped Time per Epoch v.s. Number of GPUs')
#         plt.xlabel('Number of GPUs')
#         plt.ylabel('Average Elapsed Time per Epoch in Seconds')
#         plt.legend()
#         plt.savefig(f'plots/emb{phase}runtime_vs_num_gpus_bar.png')
#         plt.clf()

def plot_transformer_layer_runtime_vs_hidden_sizes():
    for nheads in NUM_ATTENTION_HEADS:
        for phase in PHASES:
            for num_gpu in NUM_GPUS:
                runtimes = [get_transformer_layer_runtime(phase, num_gpu, hidden_size, nheads, 8) for hidden_size in HIDDEN_SIZES]
                plt.scatter(HIDDEN_SIZES, runtimes, label=f'{num_gpu} GPUs')
                plt.plot(HIDDEN_SIZES, runtimes)
            phase_title = phase.get_pretty_name()
            plt.title(f'Transformer Layer {phase_title}:\nNumber of Self-attention Heads = {nheads}\nAverage Elasped Time per Pass v.s. Hidden Size (Using {NUM_GPUS[-1]} GPU)')
            plt.xlabel('Hidden Size')
            plt.ylabel('Average Elasped Time per Pass in Seconds')
            plt.legend()
            plt.savefig(f'plots/transformer_layer{phase}nah-{nheads}_runtime_vs_num_heads.png')
            plt.clf()

def plot_transformer_layer_runtime_vs_num_gpus():
    for nheads in NUM_ATTENTION_HEADS:
        for phase in PHASES:
            for hidden_size in HIDDEN_SIZES:
                runtimes = [get_transformer_layer_runtime(phase, num_gpu, hidden_size, nheads, 8) for num_gpu in NUM_GPUS]
                plt.scatter(NUM_GPUS, runtimes, label=f'Hidden Size {hidden_size}')
                plt.plot(NUM_GPUS, runtimes)
            phase_title = phase.get_pretty_name()
            plt.title(f'Transformer Layer {phase_title}:\n'
                      f'Number of Self-attention Heads = {nheads}\n'
                      f'Average Elasped Time per Pass v.s. Number of GPUs')
            plt.xlabel('Number of GPUs')
            plt.ylabel('Average Elasped Time per Pass in Seconds')
            plt.legend()
            plt.savefig(f'plots/transformer_layer{phase}nah-{nheads}_runtime_vs_num_gpus.png')
            plt.clf()

def plot_gpt2_runtime_vs_num_params():
    for phase in PHASES:
        for num_gpu in NUM_GPUS:
            runtime_list = []
            nparams_list = []
            for nparams, np_hs_nheads_tuple in NPARAM_2_NP_HS_NHEADS.items():
                runtime = get_gpt2_runtime(phase, num_gpu, np_hs_nheads_tuple[1], np_hs_nheads_tuple[2], 8)
                if runtime:
                    runtime_list.append(runtime)
                    nparams_list.append(nparams)
            plt.scatter(nparams_list, runtime_list, label=f'Number of GPUs {num_gpu}')
            plt.plot(nparams_list, runtime_list)
        phase_title = "Forward + Backward Pass" if phase == Phase.TOTAL else phase.get_pretty_name()
        plt.title(f'GPT2 {phase_title}:\nAverage Elasped Time per Pass v.s. Number of Learnable Parameters \n(Using {NUM_GPUS[-1]} GPU)')
        plt.xlabel('Number of Learnable Parameters')
        plt.ylabel('Average Elasped Time per Pass in Seconds')
        plt.legend()
        plt.savefig(f'plots/gpt2{phase}_runtime_vs_num_params.png')
        plt.clf()
                

def area_plot_runtime_vs_num_gpus(ys, xs, model, yname, xname):
    if model == 'transformer layer':
        get_runtime_func = get_transformer_layer_runtime
    elif model == 'gpt2':
        get_runtime_func = get_gpt2_runtime
    elif model == 'embedding':
        get_runtime_func = get_emb_runtime
    else:
        raise ValueError('unrecognized model')
    fig, axes = plt.subplots(nrows=len(ys), ncols=len(xs))
    # fig.tight_layout()
    for i, nheads in enumerate(ys):
        for j, hidden_size in enumerate(xs):
            forward = []
            backward = []
            num_gpus = []
            for num_gpu in NUM_GPUS:
                fout = get_runtime_func(Phase.FORWARD, num_gpu, hidden_size, nheads, 8)
                bout = get_runtime_func(Phase.BACKWARD, num_gpu, hidden_size, nheads, 8)
                if fout != None and bout != None:
                    # assert bout != None
                    forward.append(fout)
                    backward.append(bout)
                    num_gpus.append(num_gpu)
            df = pd.DataFrame({
                'forward': forward,
                'backward': backward,
            }, index=num_gpus)

            ylabel = str(ys[i]) if j == 0 else None
            xlabel = str(xs[j]) if i == len(ys)-1 else None
            
            # ylim = (0,1.3)
            ylim = (0,0.8)
            figsize=(15,10)

            # if i < len(ys)-1:
            #     if j > 0:
            #         df.plot.area(ax=axes[i,j], legend=None, figsize=figsize, xticks=[], yticks=[], ylabel=ylabel, xlabel=xlabel, ylim=ylim)
            #     else:
            #         df.plot.area(ax=axes[i,j], legend=None, figsize=figsize, xticks=[], ylabel=ylabel, xlabel=xlabel, ylim=ylim)
            # else:
            #     if j > 0:
            #         df.plot.area(ax=axes[i,j], legend=None, figsize=figsize, xticks=NUM_GPUS, yticks=[], ylabel=ylabel, xlabel=xlabel, ylim=ylim)
            #     else:
            #         df.plot.area(ax=axes[i,j], legend=None, figsize=figsize, xticks=NUM_GPUS, ylabel=ylabel, xlabel=xlabel, ylim=ylim)

            if i < len(ys)-1:
                df.plot.area(ax=axes[i,j], legend=None, figsize=(20,10), xticks=[], ylabel=ylabel, xlabel=xlabel, ylim=ylim)
            else:
                df.plot.area(ax=axes[i,j], legend=None, figsize=(20,10), xticks=NUM_GPUS, ylabel=ylabel, xlabel=xlabel, ylim=ylim)
        
    fig.suptitle(f'{model}:\n'
                f'Average Elasped Time per Pass v.s. Number of GPUs')
    fig.text(0.5, 0.04, xname, ha='center')
    fig.text(0.04, 0.5, yname, va='center', rotation='vertical')

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')

    plt.savefig(f'plots/area_{model}_runtime_vs_num_gpus2.png')
    plt.clf()

def area_plot_runtime_vs_num_gpus_emb(xs, model, xname):
    get_runtime_func = get_emb_runtime
    fig, axes = plt.subplots(nrows=1, ncols=len(xs))
    fig.subplots_adjust(bottom=0.2)

    for j, vocab_size in enumerate(xs):
        forward = []
        backward = []
        num_gpus = []
        for num_gpu in NUM_GPUS:
            fout = get_runtime_func(Phase.FORWARD, num_gpu, vocab_size)
            bout = get_runtime_func(Phase.BACKWARD, num_gpu, vocab_size)
            if fout != None and bout != None:
                # assert bout != None
                forward.append(fout)
                backward.append(bout)
                num_gpus.append(num_gpu)
        df = pd.DataFrame({
            'forward': forward,
            'backward': backward,
        }, index=num_gpus)

        # ylabel = str(ys[i]) if j == 0 else None
        xlabel = str(xs[j])
        
        ylim = (0,0.04)
        # ylim = (0,0.8)

        df.plot.area(ax=axes[j], legend=None, figsize=(20,2.5), xticks=NUM_GPUS, xlabel=xlabel, ylim=ylim)
        
    fig.suptitle(f'{model}: Average Elasped Time per Pass v.s. Number of GPUs\n')
    fig.text(0.5, 0.02, xname, ha='center')
    # fig.text(0.04, 0.5, yname, va='center', rotation='vertical')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')

    plt.savefig(f'plots/area_{model}_runtime_vs_num_gpus.png', pad_inches=10)
    plt.clf()

def initialize_nparams_2_np_hs_nheads():
    import os
    import re
    filenames = os.listdir('./benchmark/exp')
    gpt2_files = [filename for filename in filenames if 'gpt2' in filename]
    nparams_np_hs_nah_tuple_list = []
    for gpt2_file in gpt2_files:
        splitted = re.split('-|_', gpt2_file)
        if 'forward' in gpt2_file or 'backward' in gpt2_file:
            nparams_np_hs_nah_tuple = (int(splitted[-1]), int(splitted[3]), int(splitted[5]), int(splitted[9]))
        else:
            nparams_np_hs_nah_tuple = (int(splitted[-1]), int(splitted[2]), int(splitted[4]), int(splitted[8]))
        nparams_np_hs_nah_tuple_list.append(nparams_np_hs_nah_tuple)
    nparams_np_hs_nah_tuple_list = sorted(set(nparams_np_hs_nah_tuple_list), 
                                          key=lambda nparams_hs_nah_tuple: nparams_hs_nah_tuple[0] * nparams_hs_nah_tuple[1])
    nparams_2_np_hs_nheads = OrderedDict()
    for nparam, np, hs, nheads in nparams_np_hs_nah_tuple_list:
        nparams_2_np_hs_nheads[nparam * np] = (np, hs, nheads)
    global NPARAM_2_NP_HS_NHEADS
    NPARAM_2_NP_HS_NHEADS = nparams_2_np_hs_nheads

if __name__ == '__main__':
    # average_elapsed_time_vs_model_size_all()
    # average_elapsed_time_vs_num_gpus_all()
    # get_emb_runtime(Phase.BACKWARD, 1, 1024)
    # plot_emb_runtime_vs_num_gpus()
    # plot_emb_runtime_vs_vocab_size()
    # plot_emb_runtime_vs_num_gpus()
    # plot_transformer_layer_runtime_vs_hidden_sizes()
    # plot_transformer_layer_runtime_vs_num_gpus()
    # initialize_nparams_2_np_hs_nheads()
    # plot_gpt2_runtime_vs_num_params()
    # plot_emb_runtime_vs_vocab_size()
    # plot_emb_runtime_vs_num_gpus()
    mlp_average_elapsed_time_vs_model_size_all()
    mlp_average_elapsed_time_vs_num_gpus_all()

    # area_plot_runtime_vs_num_gpus(NUM_ATTENTION_HEADS, HIDDEN_SIZES, 'transformer layer', 'num attention heads', 'hidden size')
    # area_plot_runtime_vs_num_gpus(NUM_ATTENTION_HEADS, HIDDEN_SIZES, 'gpt2', 'num attention heads', 'hidden size')
    # area_plot_runtime_vs_num_gpus_emb(VOCAB_SIZES, 'embedding', 'vocab size')
    pass