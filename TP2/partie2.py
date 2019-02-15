import torch
import torchvision
import torchvision.datasets as datasets
from cnn import CNN
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

print ("TP2 INF8225..")

# Source : https://mc.ai/intro-to-pytorch-with-image-classification-on-a-fashion-clothes-dataset/

# Global setup
nb_epoch = 5
batch_size = 80
lr = 0.001

# Download the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
mnist_trainset = datasets.MNIST(root="./mnist", train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root="./mnist", train=False, download=True, transform=transform)

# Load the dataset
train_loader = torch.utils.data.DataLoader(dataset=mnist_trainset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_testset,batch_size=batch_size,shuffle=True)

# Instantiate the cnn
cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)



# Training
losses = []
for epoch in range(nb_epoch):
    print("epoch " + str(epoch) + "...")
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())


# Testing
correct = 0
for images, labels in test_loader:
    outputs = cnn(images)
    predict = torch.argmax(outputs, 1)
    correct = correct + (predict == labels).sum()
    

print ("Precision sur le test set : ", correct.item() / test_loader.sampler.num_samples * 100)



# Plot
plt.plot(losses[0::500])
plt.show()

print ("DONE")