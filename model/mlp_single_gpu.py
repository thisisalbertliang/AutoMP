import torch

class MLPSingleGpu(torch.nn.Module):

    def __init__(self, num_features, num_classes, hidden_sizes):
        super(MLPSingleGpu, self).__init__()

        assert len(hidden_sizes) > 0, 'AutoMP: Must have at least 1 hidden layer'

        # self.fc = []
        # self.relu = []
        # for idx, prev_hidden_size, hidden_size in enumerate(zip([num_features] + hidden_sizes, hidden_sizes)):
        #     fc = torch.nn.Linear(in_features=prev_hidden_size, out_features=hidden_size, bias=True)
        #     self.fc.append(fc)
        #     relu = torch.nn.ReLU()
        #     self.relu.append(relu)
        #     self.register_parameter(name=f'fc{idx}', param=fc)
        #     self.register_parameter(relu)
        # self.fc_final = torch.nn.Linear(in_features=hidden_sizes[-1], out_features=num_classes, bias=True)
        self.fc1 = torch.nn.Linear(in_features=num_features, out_features=hidden_sizes[0], bias=True)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=hidden_sizes[0],out_features=hidden_sizes[1], bias=True)
        self.relu2 = torch.nn.ReLU()
        # self.fc3 = torch.nn.Linear(in_features=hidden_sizes[1],out_features=hidden_sizes[2], bias=True)
        # self.relu3 = torch.nn.ReLU()
        self.fc_final = torch.nn.Linear(in_features=hidden_sizes[1], out_features=num_classes, bias=True)

    def forward(self, X):
        # for fc, relu in zip(self.fc, self.relu):
        #     X = fc.forward(X)
        #     X = relu.forward(X)
        # logits = self.fc_final.forward(X)
        X = self.fc1.forward(X)
        X = self.relu1.forward(X)
        X = self.fc2.forward(X)
        X = self.relu2.forward(X)
        # X = self.fc3.forward(X)
        # X = self.relu3.forward(X)
        logits = self.fc_final.forward(X)
        return logits
