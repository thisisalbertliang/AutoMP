import torch
import torch.distributed
import numpy as np
import time
from arguments import parse_args, get_args
from model.mlp_single_gpu import MLPSingleGpu


def train(hidden_sizes, device):
    # Parse command line arguments
    parse_args()
    # Set CUDA device
    torch.cuda.set_device(device)

    # Use MNIST data
    train_data = np.genfromtxt('data/digitstrain.txt', delimiter=",")
    train_X = torch.tensor(train_data[:, :-1], dtype=torch.float, device=torch.cuda.current_device())
    train_Y = torch.tensor(train_data[:, -1], dtype=torch.int64, device=torch.cuda.current_device())
    print(f'AutoMP: train_X shape: {train_X.size()}')
    print(f'AutoMP: train_Y shape: {train_Y.size()}')

    num_features = train_X.size()[1]
    num_classes = 10
    assert num_features == 28*28
    mlp = MLPSingleGpu(num_features=num_features, num_classes=num_classes, hidden_sizes=hidden_sizes)
    mlp.cuda(torch.cuda.current_device())
    print('AutoMP: Successfully initialized MLPSingleGPU')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01)

    num_epochs = 500
    num_train_samples = train_X.size()[0]
    batch_size = num_train_samples
    prev_time = time.time()
    for epoch in range(num_epochs):
        if epoch > 0:
            print(f' Elapsed time: {time.time()-prev_time}')
            prev_time = time.time()
        # start_time = time.time()
        train_loss = 0
        for sample_idx in range(0, num_train_samples, batch_size):
            mini_batch = train_X[sample_idx:sample_idx+batch_size, ...]
            labels = train_Y[sample_idx:sample_idx+batch_size]
            # Forward pass
            logits = mlp(mini_batch)
            # Note: torch.nn.CrossEntropyLoss does not need one hot encoding
            loss = criterion(logits, labels)
            train_loss += loss
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= (num_train_samples / batch_size)
        # end_time = time.time()
        # print('b')
        # if epoch % 50 == 0:
        print(f'Epoch Number {epoch}: train loss: {train_loss}')


if __name__ == '__main__':
    train([20480,20480,20480], device=1)