import torch

class MLPSingleGpu(torch.nn.Module):

    def __init__(self, num_features, num_classes, hidden_sizes=[20480,20480,20480]):
        super(MLPSingleGpu, self).__init__()

        self.fc1 = torch.nn.Linear(in_features=num_features, out_features=hidden_sizes[0], bias=True)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=hidden_sizes[0],out_features=hidden_sizes[1], bias=True)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(in_features=hidden_sizes[1],out_features=hidden_sizes[2], bias=True)
        self.relu3 = torch.nn.ReLU()
        self.fc_final = torch.nn.Linear(in_features=hidden_sizes[2], out_features=num_classes, bias=True)

    def forward(self, X):
        X = self.fc1.forward(X)
        X = self.relu1.forward(X)
        X = self.fc2.forward(X)
        X = self.relu2.forward(X)
        X = self.fc3.forward(X)
        X = self.relu3.forward(X)
        logits = self.fc_final.forward(X)
        return logits
