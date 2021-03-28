import torch
from model.linear import ParallelLinear


class ParallelMLP(torch.nn.Module):

    def __init__(self, num_features, num_classes, hidden_sizes=[20480,20480,20480]):
        super(ParallelMLP, self).__init__()

        self.fc1_relu = ParallelLinear(input_size=num_features, output_size=hidden_sizes[0], gather=True)
        self.fc2_relu = ParallelLinear(input_size=hidden_sizes[0], output_size=hidden_sizes[1], gather=True)
        # self.fc3_relu = ParallelLinear(input_size=hidden_sizes[1], output_size=hidden_sizes[2], gather=True)
        self.fc_final = ParallelLinear(input_size=hidden_sizes[1], output_size=num_classes, gather=True)

    def forward(self, X):
        X = self.fc1_relu.forward(X)
        X = self.fc2_relu.forward(X)
        # X = self.fc3_relu.forward(X)
        logits = self.fc_final.forward(X)
        return logits
