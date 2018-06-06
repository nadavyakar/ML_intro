from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms_
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler


data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

valid_ratio = .2
batch_size = 128
num_workers = 128
nclasses = 10
nepocs = 1

image_dataset = datasets.CIFAR10('.',transform=data_transform, download=True)
image_test = datasets.CIFAR10('.',transform=data_transform, train=False, download=True)

dataset_size = len(image_dataset.train_data)
train_size = int(dataset_size*(1-valid_ratio))
dataset_sizes = {'train': train_size, 'val' : dataset_size-train_size}

indices = list(range(dataset_size))

validation_idx = np.random.choice(indices, size=dataset_sizes['val'], replace=False)
train_idx = list(set(indices) - set(validation_idx))

dataloaders = {'train': torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
                                                    sampler=SubsetRandomSampler(train_idx),
                                                    num_workers=num_workers),
               'val': torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
                                                    sampler=SubsetRandomSampler(validation_idx),
                                                    num_workers=num_workers),
               'test': torch.utils.data.DataLoader(image_test, batch_size=batch_size,
                                                   num_workers=num_workers)}
device = "cpu"
inputs = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs[0])

def train_model(dataloaders, model, mode_name, criterion, optimizer, scheduler, num_epochs):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_losses = {'train':[], 'val':[]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            epoch_losses[phase].append(epoch_loss)

            print('model:{} phase:{} Loss: {:.4f} Acc: {:.4f}'.format(mode_name, phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_losses

def test(dataloader_test, model):
    running_loss = 0.0
    running_corrects = 0
    label_list = []
    pred_list = []
    for inputs, labels in dataloader_test:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        pred_list+=[x.item() for x in preds]
        label_list+=list(labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    testset_size = len(dataloader_test)
    return label_list, pred_list, running_loss / testset_size, running_corrects.double() / testset_size


# ResNet as fixed feature extractor
mode_name = "resnet-18"
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, nclasses)
model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
model_conv, epoch_losses = train_model(dataloaders, model_conv, mode_name, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=nepocs)
# plot resett-18 train and val loss
epocs_list=range(nepocs)
plt.plot(epocs_list, epoch_losses['train'], 'red', epocs_list, epoch_losses['val'], 'green', linewidth=1, markersize=1)
plt.xlabel("epocs")
plt.savefig("loss.model_{}.phase_train_and_val.png".format(mode_name))
plt.clf()
# print final resnet-18 loss & acc
label_list, pred_list, epoch_loss, epoch_acc = test(dataloader_test, model_conv)
print('model:{} phasse:{} Loss: {:.4f} Acc: {:.4f}'.format(mode_name,"test", epoch_loss, epoch_acc))
print("confusion matrix of {}:\n{}".format(mode_name,confusion_matrix(label_list, pred_list)))