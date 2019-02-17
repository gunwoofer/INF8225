import torch
import torchvision
import torch.nn as nn

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(
            576, 
            32,
            bias=True
        )

        self.fc2 = nn.Linear(
            32, 
            10,
            bias=True
        )
    
    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

