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
class ConvSmallCIFAR10(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(ConvSmallCIFAR10, self).__init__()
        # First block of convolutions
        self.conv1 = nn.Conv2d(dim_input, 96, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.5)

        # Second block of convolutions
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.5)

        # Third block of convolutions
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1)
        self.conv9 = nn.Conv2d(192, 192, kernel_size=1)

        # Global average pooling and dense layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, dim_output)

    def forward(self, x):
        # Apply first block of convolutions
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)

        # Apply second block of convolutions
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = self.maxpool2(x)
        x = self.dropout2(x)

        # Apply third block of convolutions
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))
        x = F.leaky_relu(self.conv9(x))

        # Global average pooling and dense layer
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)

        return x

class ConvSmallSVHN(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(ConvSmallSVHN, self).__init__()
        # First block of convolutions
        self.conv1 = nn.Conv2d(dim_input, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.5)

        # Second block of convolutions
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.5)

        # Third block of convolutions
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=1)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=1)

        # Global average pooling and dense layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, dim_output)

    def forward(self, x):
        # Apply first block of convolutions
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)

        # Apply second block of convolutions
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = self.maxpool2(x)
        x = self.dropout2(x)

        # Apply third block of
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))
        x = F.leaky_relu(self.conv9(x))

        # Global average pooling and dense layer
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)

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