import numpy as np
import torch
from pytorch_base_network import PyTorchBaseNetwork
from torch import nn
from torch import optim
import torch.nn.functional as F


import helper

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# Download and load the test data
testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


# Grab some data and show it
dataiter = iter(trainloader)
images, labels = dataiter.next()
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');

# Flatten the data for it to be fed to the fully connected neural network
# data will be a 1D vector, new images shape is (batch size, color channels, image pixels)
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to not automatically get batch size
# Forward pass through the network

#select the first image
img_idx = 0
img = images[img_idx, :]

"""

# build the default pytorch base network
model_def = PyTorchBaseNetwork()
#pass the first image trough
ps = model_def.forward(img)
#classify (before training, won't work)
helper.view_classify(img.view(1, 28, 28), ps)


#alternatively, a sequential network can be built in place:

model_seq = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(784,200)),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(200, 100)),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(100,10)),
                      ('softmax', nn.Softmax(dim=1))]))

ps = model_seq.forward(img)
helper.view_classify(img.view(1, 28, 28), ps)

"""

#as asked in the exercise, build a different network with 4 layers (3 inner +output)
model=PyTorchBaseNetwork(784, None, [100,50],10)
#classify (before training, won't work)
from collections import OrderedDict
model_seq = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(784,200)),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(200, 100)),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(100,10))]))

ps = model_seq.forward(img)
helper.view_classify(img.view(1, 28, 28), ps)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_seq.parameters(), lr=0.01)

print('Initial weights - ', model_seq.fc1.weight)

images, labels = next(iter(trainloader))
images.resize_(64, 784)

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Forward pass, then backward pass, then update weights
output = model_seq.forward(images)
loss = criterion(output, labels)
loss.backward()
print('Gradient -', model_seq.fc1.weight.grad)
optimizer.step()
print('Updated weights - ', model_seq.fc1.weight)




# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# Download and load the test data
testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)



