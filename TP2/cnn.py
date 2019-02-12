import torch
import torchvision
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)            
        )

        self.full_connected = nn.Linear(
            32, 
            10,
            bias=True
        )
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = x.view(x.size(0), -1)
        x = self.full_connected(x)
        return x

