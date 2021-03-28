import torch
import torch.distributed
import numpy as np
import time
from model.mlp import ParallelMLP
from arguments import parse_args, get_args
from model.mlp_single_gpu import MLPSingleGpu

ONE_MB = 1048576


# def train(hidden_sizes):
#     # Initialize torch.distributed
#     init_distributed()
#
#     print_rank_0('AutoMP: training MLP...')
#
#     # Use MNIST data
#     train_data = np.genfromtxt('data/digitstrain.txt', delimiter=",")
#     train_X = torch.tensor(train_data[:, :-1], dtype=torch.float, device=torch.cuda.current_device())
#     train_Y = torch.tensor(train_data[:, -1], dtype=torch.int64, device=torch.cuda.current_device())
#     print(f'AutoMP: train_X shape: {train_X.size()}')
#     print(f'AutoMP: train_Y shape: {train_Y.size()}')
#
#     num_features = train_X.size()[1]
#     num_classes = 10
#     assert num_features == 28*28
#     mlp = ParallelMLP(num_features=num_features, num_classes=num_classes, hidden_sizes=hidden_sizes)
#     mlp.cuda(torch.cuda.current_device())
#     print('AutoMP: Successfully initialized MLPSingleGPU')
#
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01)
#
#     num_epochs = 500
#     num_train_samples = train_X.size()[0]
#     batch_size = num_train_samples
#     prev_time = time.time()
#     for epoch in range(num_epochs):
#         if epoch > 0:
#             print(f' Elapsed time: {time.time()-prev_time}')
#             prev_time = time.time()
#         # start_time = time.time()
#         train_loss = 0
#         for sample_idx in range(0, num_train_samples, batch_size):
#             mini_batch = train_X[sample_idx:sample_idx+batch_size, ...]
#             labels = train_Y[sample_idx:sample_idx+batch_size]
#             # Forward pass
#             logits = mlp(mini_batch)
#             # Note: torch.nn.CrossEntropyLoss does not need one hot encoding
#             loss = criterion(logits, labels)
#             train_loss += loss
#             print(f'torch.cuda.memory_reserved({torch.cuda.current_device()}): {torch.cuda.memory_reserved(torch.cuda.current_device()) / ONE_MB}MB')
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         train_loss /= (num_train_samples / batch_size)
#         # end_time = time.time()
#         # print('b')
#         # if epoch % 50 == 0:
#         print(f'Epoch Number {epoch}: train loss: {train_loss}')


def train(hidden_sizes, num_epochs=50):

    # Initialize torch.distributed
    init_distributed()

    print_rank_0('AutoMP: training MLP...')
    # Use MNIST data
    train_data = np.genfromtxt('data/digitstrain.txt', delimiter=",")
    train_X = torch.tensor(train_data[:, :-1], dtype=torch.float, device=torch.cuda.current_device())
    train_Y = torch.tensor(train_data[:, -1], dtype=torch.int64, device=torch.cuda.current_device())
    print_rank_0(f'train_X shape: {train_X.size()}')
    print_rank_0(f'train_Y shape: {train_Y.size()}')

    num_features = train_X.size()[1]
    num_classes = 10
    assert num_features == 28*28
    mlp = ParallelMLP(num_features=num_features, num_classes=num_classes, hidden_sizes=hidden_sizes)
    print_rank_0('AutoMP: Successfully initialized ParallelMLP')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01)

    # num_epochs = 500
    num_train_samples = train_X.size()[0]
    batch_size = num_train_samples
    tot_time = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0
        for sample_idx in range(0, num_train_samples, batch_size):
            mini_batch = train_X[sample_idx:sample_idx+batch_size, ...]
            labels = train_Y[sample_idx:sample_idx+batch_size]
            # Forward pass
            logits = mlp(mini_batch)
            # Note: torch.nn.CrossEntropyLoss does not need one hot encoding
            loss = criterion(logits, labels)
            # loss = parallel_cross_entropy(logits, labels)
            train_loss += loss
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= (num_train_samples / batch_size)
        # if epoch % 50 == 0:
        print_rank_0(f'Epoch Number {epoch}: train loss: {train_loss}, time: {time.time()-start_time}')
        tot_time += time.time()-start_time
    print_rank_0(f'!!! AVG EPOCH TIME: {tot_time/num_epochs}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='training arguments')
    # parser.add_argument('--hidden-sizes', nargs='+', required=False, type=int)
    args = parser.parse_args()

    train([31952, 31952])
