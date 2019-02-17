import torch
import torchvision
import torchvision.datasets as datasets
from cnn import CNN
from fullconnected import FullConnected
from cnn2 import CNN2
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

print ("TP2 INF8225..")

# Global setup
nb_epoch = 5
batch_size = 80
lr = 0.001
best_model = None

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
mnist_trainset = datasets.MNIST(root="./mnist", train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root="./mnist", train=False, download=True, transform=transform)

# Split training and validation dataset
train_size = int(0.8 * len(mnist_trainset))
validation_size = len(mnist_trainset) - train_size
mnist_trainset, mnist_validset = torch.utils.data.random_split(mnist_trainset, [train_size, validation_size])

# Load the loaders
train_loader = torch.utils.data.DataLoader(dataset=mnist_trainset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_testset,batch_size=batch_size,shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=mnist_validset,batch_size=batch_size,shuffle=True)


# Instantiate the cnn
cnn = CNN()
# cnn = FullConnected()
# cnn = CNN2()
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)



# Training
losses = []
best_accuracy = 0
for epoch in range(nb_epoch):
    print("epoch " + str(epoch+1) + "...")
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    losses.append(loss.item() + 1)
    # Validation
    correct = 0
    for images, labels in valid_loader:
        outputs = cnn(images)
        predict = torch.argmax(outputs, 1)
        correct = correct + (predict == labels).sum()
    current_accuracy = correct.item() / valid_loader.sampler.num_samples * 100
    print("Current accuracy : ", current_accuracy)
    print("Best accuracy yet : ", best_accuracy)
    if (current_accuracy > best_accuracy):
        best_accuracy = current_accuracy
        best_model = cnn
    



# Testing
correct = 0
for images, labels in test_loader:
    outputs = best_model(images)
    predict = torch.argmax(outputs, 1)
    correct = correct + (predict == labels).sum()

print ("Accuracy on unseen data : ", correct.item() / test_loader.sampler.num_samples * 100)



# Plot
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(losses)
plt.show()

print ("DONE")