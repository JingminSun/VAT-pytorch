import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(Net, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.conv1 = nn.Conv2d(dim_input, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.fc1 = nn.Linear(128, dim_output)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, 128)
        x = self.fc1(x)
        return x

class SimpleNet(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(SimpleNet, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.conv1 = nn.Linear(dim_input, 64, bias=True)
        self.conv2 = nn.Linear(64, 128, bias=False)
        self.conv3 = nn.Linear(128, 128, bias=False)
        self.fc1 = nn.Linear(128, dim_output)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x =  F.relu(self.conv2(x))
        x =  F.relu(self.conv3(x))
        x = x.view(-1, 128)
        x = self.fc1(x)
        return x