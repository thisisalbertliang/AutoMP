import matplotlib.pyplot as plt
from enum import Enum
from statistics import mean
import numpy as np
import pandas as pd

VOCAB_SIZES = [1024, 3072, 8192, 32768, 65536, 131072, 262144, 524288]
NUM_GPUS = [1, 2, 4, 8]
HIDDEN_SIZES = [256, 512, 1024, 1536, 1920, 2304, 3072, 4096]
NUM_ATTENTION_HEADS = [8, 16, 32, 64]

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
            return 'Forward Phase'
        elif self == Phase.BACKWARD:
            return 'Backward Phase'
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


def average_elapsed_time_vs_model_size_all():
    plt.scatter(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_1_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_1_GPU)
    plt.plot(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_1_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_1_GPU, label='1 GPU')
    plt.scatter(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_2_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_2_GPU)
    plt.plot(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_2_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_2_GPU, label='2 GPU')
    plt.scatter(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_4_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_4_GPU)
    plt.plot(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_4_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_4_GPU, label='4 GPU')
    plt.scatter(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_8_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_8_GPU)
    plt.plot(MODEL_SIZES_IN_MILLIONS[:len(AVG_ELAPSED_TIME_PER_EPOCH_8_GPU)], AVG_ELAPSED_TIME_PER_EPOCH_8_GPU, label='8 GPU')
    plt.title(f'Average Elapsed Time per Epoch v.s. Model Size')
    plt.xlabel('Model Size (Number of Parameters in Millions)')
    plt.ylabel('Average Elapsed Time per Epoch in Seconds')
    plt.legend()
    plt.savefig('plots/avg_elasped_time_vs_model_size.png')
    plt.show()


def average_elapsed_time_vs_num_gpus_all():
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
    plt.title(f'Average Elapsed Time per Epoch v.s. Number of GPUs')
    plt.xlabel('Number of GPUs')
    plt.ylabel('Average Elapsed Time per Epoch in Seconds')
    plt.legend()
    plt.savefig('plots/avg_elapsed_time_vs_num_gpus2.png')
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
                     num_attention_heads: int, batch_size: int, 
                     num_params: int):
    if phase == Phase.TOTAL:
        filename = f'benchmark/exp/gpt2_np-{nproc}_hs-{hidden_size}_nah-{num_attention_heads}_bsz-{batch_size}_num-params-{num_params}'
    else:
        filename = f'benchmark/exp/gpt2{phase}np-{nproc}_hs-{hidden_size}_nl-3_nah-{num_attention_heads}_bsz-{batch_size}_num-params-{num_params}'
    with open(filename, 'r') as f:
        runtime_list = [float(l) for l in f.read().splitlines()][1:]
        mean_runtime = mean(runtime_list)
        return mean_runtime


def plot_emb_runtime_vs_vocab_size():
    for phase in PHASES:
        for num_gpu in NUM_GPUS[:1]:
            runtimes = [get_emb_runtime(phase, num_gpu, vs) for vs in VOCAB_SIZES]
            plt.scatter(VOCAB_SIZES, runtimes, label=f'{num_gpu} GPU')
        phase_title = phase.get_pretty_name()
        plt.title(f'Embedding Layer {phase_title}: Average Elasped Time per Epoch v.s. Vocab Size (Using {NUM_GPUS[-1]} GPU)')
        plt.xlabel('Vocab Size')
        plt.ylabel('Average Elasped Time per Epoch in Seconds')
        plt.legend()
        plt.savefig(f'plots/emb{phase}runtime_vs_vocab_size.png')
        plt.clf()

def plot_emb_runtime_vs_num_gpus():
    for phase in PHASES:
        for vs in VOCAB_SIZES:
            runtimes = [get_emb_runtime(phase, num_gpu, vs) for num_gpu in NUM_GPUS]
            plt.scatter(NUM_GPUS, runtimes, label=f'Vocab Size {vs}')
        phase_title = phase.get_pretty_name()
        plt.title(f'Embedding Layer {phase_title}: Average Elasped Time per Epoch v.s. Number of GPUs')
        plt.xlabel('Number of GPUs')
        plt.ylabel('Average Elapsed Time per Epoch in Seconds')
        plt.legend()
        plt.savefig(f'plots/emb{phase}runtime_vs_num_gpus.png')
        plt.clf()
        
def plot_emb_runtime_vs_num_gpus_bar():
    bottom = None
    vs = 524288
    width = 0.4
    for i, phase in enumerate(PHASES):
        if phase == Phase.TOTAL:
            continue
        if phase == Phase.FORWARD:
            c = 'r'
        elif phase == Phase.BACKWARD:
            c = 'g'    
        runtimes = [get_emb_runtime(phase, num_gpu, vs) for num_gpu in NUM_GPUS]
        print(runtimes)
        print(bottom)
        if i != 0:
            plt.bar(NUM_GPUS, runtimes, width=width, bottom=bottom, color=c)
            bottom += np.array(runtimes)
        else:
            plt.bar(NUM_GPUS, runtimes, width=width, color=c)
            bottom = np.array(runtimes)
        
    # phase_title = phase.get_pretty_name()
    plt.title(f'Embedding Layer: Average Elasped Time per Epoch v.s. Number of GPUs')
    plt.xlabel('Number of GPUs')
    plt.ylabel('Average Elapsed Time per Epoch in Seconds')
    plt.legend()
    plt.savefig(f'plots/emb{phase}runtime_vs_num_gpus_bar.png')
    plt.clf()

def area_plot_transformer_layer_runtime_vs_num_gpus():
    # for nheads in NUM_ATTENTION_HEADS:
    nheads = 8
    hidden_size = HIDDEN_SIZES[-1]

    fig, axes = plt.subplots(nrows=len(NUM_ATTENTION_HEADS), ncols=len(HIDDEN_SIZES))
    for i, nheads in enumerate(NUM_ATTENTION_HEADS):
        for j, hidden_size in enumerate(HIDDEN_SIZES):
            df = pd.DataFrame({
                'forward': [get_transformer_layer_runtime(Phase.FORWARD, num_gpu, hidden_size, nheads, 8) for num_gpu in NUM_GPUS],
                'backward': [get_transformer_layer_runtime(Phase.BACKWARD, num_gpu, hidden_size, nheads, 8) for num_gpu in NUM_GPUS],
            }, index=NUM_GPUS)

            ylabel = str(NUM_ATTENTION_HEADS[i]) if j == 0 else None
            xlabel = str(HIDDEN_SIZES[j]) if i == len(NUM_ATTENTION_HEADS)-1 else None
            
            if i < len(NUM_ATTENTION_HEADS)-1:
                df.plot.area(ax=axes[i,j], legend=None, figsize=(20,10), xticks=[], ylabel=ylabel, xlabel=xlabel)
            else:
                df.plot.area(ax=axes[i,j], legend=None, figsize=(20,10), xticks=NUM_GPUS, ylabel=ylabel, xlabel=xlabel)
        
    fig.suptitle(f'Transformer Layer:\n'
                f'Average Elasped Time per Pass v.s. Number of GPUs')
    fig.text(0.5, 0.04, 'Hidden Size', ha='center')
    fig.text(0.04, 0.5, 'Number of Attention Heads', va='center', rotation='vertical')

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')

    # fig.text(0.5, 0.04, 'Number of GPUs', ha='center')
    # fig.text(0.04, 0.5, 'Average Elasped Time per Epoch in Seconds', va='center', rotation='vertical')

    # plt.legend()
    plt.savefig(f'plots/area_transformer_layer_nah-{nheads}_runtime_vs_num_gpus.png')
    plt.clf()

def area_plot_runtime_vs_num_gpus(ys, xs, model, yname, xname):
    if model == 'transformer layer':
        get_runtime_func = get_transformer_layer_runtime
    elif model == 'gpt2':
        get_runtime_func = get_gpt2_runtime
    else:
        raise ValueError('unrecognized model')
    fig, axes = plt.subplots(nrows=len(ys), ncols=len(xs))
    for i, nheads in enumerate(ys):
        for j, hidden_size in enumerate(xs):
            df = pd.DataFrame({
                'forward': [get_runtime_func(Phase.FORWARD, num_gpu, hidden_size, nheads, 8) for num_gpu in NUM_GPUS],
                'backward': [get_runtime_func(Phase.BACKWARD, num_gpu, hidden_size, nheads, 8) for num_gpu in NUM_GPUS],
            }, index=NUM_GPUS)

            ylabel = str(ys[i]) if j == 0 else None
            xlabel = str(xs[j]) if i == len(ys)-1 else None
            
            if i < len(ys)-1:
                df.plot.area(ax=axes[i,j], legend=None, figsize=(20,10), xticks=[], ylabel=ylabel, xlabel=xlabel)
            else:
                df.plot.area(ax=axes[i,j], legend=None, figsize=(20,10), xticks=NUM_GPUS, ylabel=ylabel, xlabel=xlabel)
        
    fig.suptitle(f'{model}:\n'
                f'Average Elasped Time per Pass v.s. Number of GPUs')
    fig.text(0.5, 0.04, xname, ha='center')
    fig.text(0.04, 0.5, yname, va='center', rotation='vertical')

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')

    plt.savefig(f'plots/area_{model}_runtime_vs_num_gpus.png')
    plt.clf()

if __name__ == '__main__':
    # average_elapsed_time_vs_model_size_all()
    # average_elapsed_time_vs_num_gpus_all()
    # get_emb_runtime(Phase.BACKWARD, 1, 1024)
    # plot_emb_runtime_vs_num_gpus()
    area_plot_runtime_vs_num_gpus(NUM_ATTENTION_HEADS, HIDDEN_SIZES, 'transformer layer', 'num attention heads', 'hidden size')
    area_plot_runtime_vs_num_gpus(NUM_ATTENTION_HEADS, HIDDEN_SIZES, 'gpt2', 'num attention heads', 'hidden size')
