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
from torchvision.datasets import CIFAR10
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms_
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(500),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.CIFAR10('.',transform=data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
image_dataset = image_datasets['train']
# image_dataset = datasets.CIFAR10('.',transform=data_transforms,download=True)
valid_ratio = .2
dataset_size = len(image_dataset.train_data)
train_size = int(dataset_size*(1-valid_ratio))
train_data = image_dataset.train_data[:train_size]
valid_data = image_dataset.train_data[train_size:]
# dataloaders = {x: torch.utils.data.DataLoader(
#     train_data if x=='train' else valid_data, batch_size=1, shuffle=True, num_workers=1) for x in ['train', 'val']}
dataset_sizes = {'train': train_size, 'val' : dataset_size-train_size}
# todo load test set
# image_dataset = datasets.CIFAR10('.',transform=data_transforms,train=False,download=True)
# dataloader_test = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=True, num_workers=1)

device = "cpu"

# Get a batch of training data
inputs = next(iter(dataloaders['train']))

#### todo ERROR: torchvision.transforms.Resize doesn't resize  input images. continue with the orig dataset instead of cifar for the meanwhile


# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
out = torchvision.utils.make_grid(inputs[0])

# imshow(out, title=[class_names[x] for x in classes])

def train_model(dataloaders, model, mode_name, criterion, optimizer, scheduler, num_epochs):
    # since = time.time()

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
                # with torch.set_grad_enabled(True):
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

    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

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

class MyModule(nn.Module):
    def __init__(self,image_size,nfc0,nfc1,nfc2):
        super(MyModule, self).__init__()
        self.image_size = image_size
        # self.fc0 = nn.Linear(image_size, nfc0)
        # self.fc1 = nn.Linear(nfc0, nfc1)
        # self.fc2 = nn.Linear(nfc1, nfc2)
        self.fc0 = nn.Linear(image_size, nfc2)
    def forward(self, x):
        # x = x.view(-1, self.image_size)
        # x = F.relu(self.fc0(x))
        # x = F.relu(self.fc1(x))
        # return F.log_softmax(self.fc2(x))
        x=np.squeeze(x.reshape(1, 1, -1))
        x = x.view(-1, self.image_size)
        return F.log_softmax(self.fc0(x))


def train_my_model(train_loader,model,optimizer):
    model.train()
    loss_sum=0
    correct = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]

        loss = F.nll_loss(output, labels)

        # loss_sum += loss.data[0]  # sum up batch loss
        loss_sum += F.nll_loss(output, labels, size_average=False).item()
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

        loss.backward()
        optimizer.step()
    return loss_sum /len(train_loader.dataset), 100. * correct / len(train_loader.dataset)

def test_and_validate(dataset_loader, model, ndataset=None, return_pred=False):
    model.eval()
    loss = 0.
    correct = 0.
    if ndataset is None:
        ndataset = len(dataset_loader.dataset)
    pred_list=[]
    for data, target in dataset_loader:
        output = model(data)
        loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        if return_pred:
            pred_list+=[x.item() for x in pred]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    return loss/ ndataset, 100. * correct / ndataset, pred_list

# nfc0=100
# nfc1=50
# nclasses=10
# nfc2=nclasses
# image_size=28*28
# nepocs=10
# valid_ratio=0.2



nepocs=1
nclasses=2 #on cifar its 10
ntrain=dataset_sizes['train']
nvalid=dataset_sizes['val']
# my model
mode_name = "my_module"
model = MyModule(224*224*3,1,1,nclasses)
lr=0.001
optimizer=optim.SGD(model.parameters(), lr=lr)

epocs_list = list(range(nepocs))
train_loss = 0.0
avg_train_loss_list = []
train_acc = 0.0
train_acc_list = []
valid_loss = 0.0
avg_valid_loss_list = []
valid_acc = 0.0
valid_acc_list = []
for e in epocs_list:
    print("epoc: {}".format(e))
    train_loss, train_acc = train_my_model(dataloaders['train'], model, optimizer)

    train_loss, train_acc, dummy_list = test_and_validate(dataloaders['train'], model, ntrain)
    avg_train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    valid_loss, valid_acc, dummy_list = test_and_validate(dataloaders['val'], model, nvalid)
    avg_valid_loss_list.append(valid_loss)
    valid_acc_list.append(valid_acc)
test_avg_loss, test_acc, pred_list = test_and_validate(dataloader_test, model, None, True)
print('Training set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(train_loss, train_acc))
print('Validation set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(valid_loss, valid_acc))
print('Test set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(test_avg_loss, test_acc))

# plt.plot(epocs_list, avg_train_loss_list, 'red', epocs_list, avg_valid_loss_list, 'green', linewidth=1, markersize=1)
# plt.xlabel("epocs")
# plt.savefig("loss.model_{}.optimizer_{}.batch_{}.lr_{}.png".format(i, optimizer_name, batch_size, lr))
# plt.clf()
# plt.plot(epocs_list, train_acc_list, 'red', epocs_list, valid_acc_list, 'green', linewidth=1, markersize=1)
# plt.xlabel("epocs")
# plt.savefig("accuracy.model_{}.optimizer_{}.batch_{}.lr_{}.png".format(i, optimizer_name, batch_size, lr))
# plt.clf()
#
# with open("test.pred", 'w') as f:
#     for pred in pred_list:
#         f.write("{}\n".format(pred))




epocs_list=range(nepocs)
plt.plot(epocs_list, epoch_losses['train'], 'red', epocs_list, epoch_losses['val'], 'green', linewidth=1, markersize=1)
plt.xlabel("epocs")
plt.savefig("loss.model_{}.phase_train_and_val.png".format(mode_name))
plt.clf()

label_list, pred_list, epoch_loss, epoch_acc = test(dataloader_test, model_conv)
print('model:{} phasse:{} Loss: {:.4f} Acc: {:.4f}'.format(mode_name,"test", epoch_loss, epoch_acc))
print("confusion matrix of {}:\n{}".format(mode_name,confusion_matrix(label_list, pred_list)))


# visualize_model(model_conv)
epocs_list=range(nepocs)
plt.plot(epocs_list, avg_train_loss_list, 'red', epocs_list, avg_valid_loss_list, 'green', linewidth=1, markersize=1)
plt.xlabel("epocs")
plt.savefig("loss.model_{}.phase_train_and_val.png".format(mode_name))
plt.clf()

# label_list, pred_list, epoch_loss, epoch_acc = test(dataloader_test, model_conv)
# print('model:{} phasse:{} Loss: {:.4f} Acc: {:.4f}'.format(mode_name,"test", epoch_loss, epoch_acc))
# print("confusion matrix of {}:\n{}".format(confusion_matrix(label_list, pred_list)))




# ResNet as fixed feature extractor
mode_name = "resnet-18"
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, nclasses)
model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
model_conv, epoch_losses = train_model(dataloaders, model_conv, mode_name, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=nepocs)

# visualize_model(model_conv)
epocs_list=range(nepocs)
plt.plot(epocs_list, epoch_losses['train'], 'red', epocs_list, epoch_losses['val'], 'green', linewidth=1, markersize=1)
plt.xlabel("epocs")
plt.savefig("loss.model_{}.phase_train_and_val.png".format(mode_name))
plt.clf()

label_list, pred_list, epoch_loss, epoch_acc = test(dataloader_test, model_conv)
print('model:{} phasse:{} Loss: {:.4f} Acc: {:.4f}'.format(mode_name,"test", epoch_loss, epoch_acc))
print("confusion matrix of {}:\n{}".format(mode_name,confusion_matrix(label_list, pred_list)))
# plt.ioff()
# plt.show()