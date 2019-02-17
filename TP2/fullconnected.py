import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class FullConnected(nn.Module):
    def __init__(self):
        super(FullConnected, self).__init__()

        self.fc1 = nn.Linear(
            28*28, 
            784,
            bias=True
        )

        self.fc2 = nn.Linear(
            784, 
            10,
            bias=True
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

