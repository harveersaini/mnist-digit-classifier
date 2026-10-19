"""CNN for normalized grayscale MNIST images; outputs log probabilities."""
import torch
from torch import nn
from torch.nn import functional as F


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(F.max_pool2d(x, 2))
        x = torch.flatten(x, 1)
        x = self.dropout2(F.relu(self.fc1(x)))
        return F.log_softmax(self.fc2(x), dim=1)
