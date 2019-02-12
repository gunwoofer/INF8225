import torch
import torchvision
import torchvision.datasets as datasets
from cnn import CNN
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable

print ("TP2 INF8225..")


# Global setup
nb_epoch = 5
batch_size = 100
lr = 0.001

# Download the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
mnist_trainset = datasets.MNIST(root="./mnist", train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root="./mnist", train=False, download=True, transform=None)

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
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.float())
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (i+1) % 100 == 0:
            print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' 
                   %(epoch+1, nb_epoch, i+1, len(mnist_trainset)//batch_size, loss.item()))
print ("DONE")