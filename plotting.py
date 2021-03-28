import matplotlib.pyplot as plt

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
    plt.scatter(x=[num_gpu for num_gpu in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_0.keys()],
                y=[avg_elapsed_time for avg_elapsed_time in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_0.values()],
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[0]}M')
    plt.scatter(x=[num_gpu for num_gpu in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_1.keys()],
                y=[avg_elapsed_time for avg_elapsed_time in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_1.values()],
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[1]}M')
    plt.scatter(x=[num_gpu for num_gpu in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_2.keys()],
                y=[avg_elapsed_time for avg_elapsed_time in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_2.values()],
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[2]}M')
    plt.scatter(x=[num_gpu for num_gpu in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_3.keys()],
                y=[avg_elapsed_time for avg_elapsed_time in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_3.values()],
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[3]}M')
    plt.scatter(x=[num_gpu for num_gpu in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_4.keys()],
                y=[avg_elapsed_time for avg_elapsed_time in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_4.values()],
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[4]}M')
    plt.scatter(x=[num_gpu for num_gpu in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_5.keys()],
                y=[avg_elapsed_time for avg_elapsed_time in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_5.values()],
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[5]}M')
    plt.scatter(x=[num_gpu for num_gpu in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_6.keys()],
                y=[avg_elapsed_time for avg_elapsed_time in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_6.values()],
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[6]}M')
    plt.scatter(x=[num_gpu for num_gpu in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_7.keys()],
                y=[avg_elapsed_time for avg_elapsed_time in NUM_GPU_2_AVG_ELAPSED_TIME_PER_EPOCH_MODEL_7.values()],
                label=f'MLP-{MODEL_SIZES_IN_MILLIONS[7]}M')
    plt.title(f'Average Elapsed Time per Epoch v.s. Number of GPUs')
    plt.xlabel('Number of GPUs')
    plt.ylabel('Average Elapsed Time per Epoch in Seconds')
    plt.legend()
    plt.savefig('plots/avg_elapsed_time_vs_num_gpus.png')
    plt.show()


if __name__ == '__main__':
    average_elapsed_time_vs_model_size_all()
    average_elapsed_time_vs_num_gpus_all()
