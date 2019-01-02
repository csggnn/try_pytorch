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
datasets_names = "MNIST", "F_MNIST"

dataset_name = "F_MNIST"

if dataset_name == "MNIST" :
    # Download and load the training data
    trainset = datasets.MNIST(dataset_name + "_data/", download=True, train=True, transform=transform)
    # Download and load the test data
    testset = datasets.MNIST(dataset_name + "_data/", download=True, train=False, transform=transform)
else:
    # Download and load the training data
    trainset = datasets.FashionMNIST(dataset_name + "_data/", download=True, train=True, transform=transform)
    # Download and load the test data
    testset = datasets.FashionMNIST(dataset_name + "_data/", download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


# from collections import OrderedDict
# model = nn.Sequential(OrderedDict([
#                       ('fc1', nn.Linear(784,200)),
#                       ('relu1', nn.ReLU()),
#                       ('fc2', nn.Linear(200, 100)),
#                       ('relu2', nn.ReLU()),
#                       ('output', nn.Linear(100,10))]))



model =PyTorchBaseNetwork(784, None, [200,100],10)




criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)


epochs = 5
print_every = 40
steps = 0
for e in range(epochs):
    running_loss = 0
    for images, labels in iter(trainloader):
        steps += 1
        # Flatten MNIST images into a 784 long vector
        images.resize_(images.size()[0], 784)

        optimizer.zero_grad()

        # Forward and backward passes
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e + 1, epochs),
                  "Loss: {:.4f}".format(running_loss / print_every))

            running_loss = 0


images, labels = next(iter(trainloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logits = model.forward(img)

# Output of the network are logits, need to take softmax for probabilities
ps = F.softmax(logits, dim=1)
helper.view_classify(img.view(1, 28, 28), ps)

print("done")