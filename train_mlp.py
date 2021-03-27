import torch
import torch.distributed
import numpy as np
import time
from initialize import init_distributed
from arguments import parse_args, get_args
from model.linear import ParallelLinear
from utils import print_rank_0
from model.cross_entropy import parallel_cross_entropy
from model.mlp import ParallelMLP


def train():
    # Parse command line arguments
    parse_args()
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
    mlp = ParallelMLP(num_features=num_features, num_classes=num_classes)
    print_rank_0('AutoMP: Successfully initialized ParallelMLP')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.001)

    num_epochs = 500
    num_train_samples = train_X.size()[0]
    batch_size = num_train_samples
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
        end_time = time.time()
        print_rank_0(f'Epoch Number {epoch}: Elapsed Time: {end_time-start_time}; train loss: {train_loss}')




    # output = parallel_linear.forward(train_X)
    # # take a look at this shit
    # print_rank_0(str(output))
    # print_rank_0(str(output.shape))
    #
    #
    # loss = parallel_cross_entropy(output, train_Y)
    # print_rank_0('loss')
    # print_rank_0(str(loss))
    # print_rank_0(str(loss.shape))


if __name__ == '__main__':
    train()