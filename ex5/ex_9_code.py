from __future__ import print_function, division

from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import copy
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import logging
logging.basicConfig(filename="nn.log",level=logging.DEBUG)

def init_data(data_transform, valid_ratio):
    # data source definition
    image_dataset = datasets.CIFAR10('.',transform=data_transform, download=True)
    image_test = datasets.CIFAR10('.',transform=data_transform, train=False, download=True)
    # division to train and valid
    dataset_size = len(image_dataset.train_data)
    train_size = int(dataset_size*(1-valid_ratio))
    dataset_sizes = {'train': train_size, 'val' : dataset_size-train_size}
    indices = list(range(dataset_size))
    validation_idx = np.random.choice(indices, size=dataset_sizes['val'], replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    # data loaders for train, valid and testing
    dataloaders = {'train': torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx),
                                                        num_workers=num_workers),
                   'val': torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(validation_idx),
                                                      num_workers=num_workers),
                   'test': torch.utils.data.DataLoader(image_test, batch_size=batch_size, num_workers=num_workers)}
    inputs = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs[0])
    return dataloaders, dataset_sizes
    # dataloaders = {'train': torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
    #                                                     sampler=SubsetRandomSampler(train_idx[:int(len(train_idx)/10)]),
    #                                                     num_workers=num_workers),
    #                'val': torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
    #                                                     sampler=SubsetRandomSampler(validation_idx[:int(len(validation_idx)/10)]),
    #                                                     num_workers=num_workers),
    #                'test': torch.utils.data.DataLoader(image_test, batch_size=batch_size,
    #                                                    sampler=SubsetRandomSampler(list(range(3000))),
    #                                                    num_workers=num_workers)}

    # dataloaders = {'train': torch.utils.data.DataLoader(image_dataset, batch_size=2,
    #                                                     sampler=SubsetRandomSampler(train_idx[:2]),
    #                                                     num_workers=2),
    #                'val': torch.utils.data.DataLoader(image_dataset, batch_size=2,
    #                                                     sampler=SubsetRandomSampler(validation_idx[:2]),
    #                                                     num_workers=2),
    #                'test': torch.utils.data.DataLoader(image_test, batch_size=2,
    #                                                    sampler=SubsetRandomSampler(list(range(2))),
    #                                                    num_workers=2)}
def train_model(dataloaders, model, mode_name, criterion, optimizer, scheduler, num_epochs, device, dataset_sizes):
    logging.debug("started training")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_losses = {'train':[], 'val':[]}
    for epoch in range(num_epochs):
        logging.debug('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
                logging.debug("training phase")
            else:
                model.eval()   # Set model to evaluate mode
                logging.debug("validation phase")
            running_loss = 0.0
            running_corrects = 0
            i=0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                i+=1
                logging.debug("example {}".format(i)) 
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
            logging.debug('model:{} phase:{} Loss: {:.4f} Acc: {:.4f}'.format(mode_name, phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_losses
def test(dataloader_test, model,criterion):
    logging.debug("testing")
    running_loss = 0.0
    running_corrects = 0
    label_list = []
    pred_list = []
    i=0
    for inputs, labels in dataloader_test:
        i+=1
        logging.debug("example {}".format(i))
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        pred_list+=[x.item() for x in preds]
        label_list+=list(labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    testset_size = len(dataloader_test.dataset)
    return label_list, pred_list, running_loss / testset_size, running_corrects.double() / testset_size
# def test_and_validate(dataset_loader, model, ndataset=None, return_pred=False):
#     model.eval()
#     if ndataset is None:
#         ndataset = len(dataset_loader.dataset)
#     for data, target in dataset_loader:
#         output = model(data)
#         loss += F.nll_loss(output, target, size_average=False).item()
#         pred = output.data.max(1, keepdim=True)[1]
#         if return_pred:
#             pred_list+=[x.item() for x in pred]
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#     return loss/ ndataset, 100. * correct / ndataset, pred_list

def run_model(mode_name, device, dataloaders, nepocs, nclasses, model_conv,optimizer_conv, dataset_sizes):
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    model_conv, epoch_losses_train_and_valid = train_model(dataloaders, model_conv, mode_name,
                                                           criterion, optimizer_conv, exp_lr_scheduler,
                                                           nepocs, device, dataset_sizes)
    label_list, pred_list, epoch_loss_test, epoch_acc_test = test(dataloaders['test'], model_conv,criterion)
    return epoch_losses_train_and_valid, epoch_loss_test, epoch_acc_test, label_list, pred_list
def run_resnet(mode_name, device, dataloaders, nepocs, nclasses, dataset_sizes):
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, nclasses)
    model_conv = model_conv.to(device)
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    return run_model(mode_name, device, dataloaders, nepocs, nclasses, model_conv,optimizer_conv, dataset_sizes)
def run_my_net(mode_name, device, dataloaders, nepocs, nclasses, dataset_sizes):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, nclasses)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # trainset = torchvision.datasets.CIFAR10(root='.', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
    #                                           shuffle=True, num_workers=2)
    # valset = torchvision.datasets.CIFAR10(root='.', train=True,
    #                                         download=True, transform=transform)
    # valloader = torch.utils.data.DataLoader(valset, batch_size=4,
    #                                           shuffle=True, num_workers=2)
    # testset = torchvision.datasets.CIFAR10(root='.', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4,
    #                                          shuffle=False, num_workers=2)
    # dataloaders ={'train':trainloader,'val':valloader,'test':testloader}
    return run_model(mode_name, device, dataloaders, nepocs, nclasses, net, optimizer, dataset_sizes)
    # net = Net()
    #
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #
    # trainset = torchvision.datasets.CIFAR10(root='.', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
    #                                           shuffle=True, num_workers=2)
    # # trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, sampler = SubsetRandomSampler(list(range(100))),
    # #                                            num_workers=1)
    #
    # testset = torchvision.datasets.CIFAR10(root='.', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4,
    #                                          shuffle=False, num_workers=2)
    # # testloader = torch.utils.data.DataLoader(testset, batch_size=1, sampler = SubsetRandomSampler(list(range(100))),
    # #                                           num_workers=1)
    #
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #
    # import torch.optim as optim
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #
    # for epoch in range(2):  # loop over the dataset multiple times
    #
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs
    #         inputs, labels = data
    #
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #
    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:  # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i + 1, running_loss / 2000))
    #             running_loss = 0.0
    #
    # print('Finished Training')
    #
    # # print images
    # # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(4):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))
    # # return epoch_losses_train_and_valid, epoch_loss_test, epoch_acc_test, label_list, pred_list
def visualize(mode_name, epoch_losses_train_and_valid, epoch_loss_test, epoch_acc_test, label_list, pred_list):
    epocs_list=range(nepocs)
    plt.plot(epocs_list, epoch_losses_train_and_valid['train'], 'red', epocs_list, epoch_losses_train_and_valid['val'], 'green', linewidth=1, markersize=1)
    plt.xlabel("epocs")
    plt.savefig("loss.model_{}.phase_train_and_val.png".format(mode_name))
    plt.clf()
    logging.debug('model:{} phasse:{} Loss: {:.4f} Acc: {:.4f}'.format(mode_name,"test", epoch_loss_test, epoch_acc_test))
    logging.debug("confusion matrix of {}:\n{}".format(mode_name,confusion_matrix(label_list, pred_list)))
# init params
valid_ratio = .2
batch_size = 32
num_workers = batch_size
nclasses = 10
nepocs = 1
device = "cpu"
# # ResNet as fixed feature extractor
# mode_name = "resnet-18_2"
# dataloaders, dataset_sizes = init_data(transforms.Compose([transforms.Resize(224), transforms.ToTensor()]), valid_ratio)
# epoch_losses_train_and_valid, epoch_loss_test, epoch_acc_test, label_list, pred_list = run_resnet(mode_name, device, dataloaders, nepocs, nclasses, dataset_sizes)
# visualize(mode_name, epoch_losses_train_and_valid, epoch_loss_test, epoch_acc_test, label_list, pred_list)
# my net
mode_name = "my net"
nepocs=1
dataloaders, dataset_sizes = init_data(transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), valid_ratio)
epoch_losses_train_and_valid, epoch_loss_test, epoch_acc_test, label_list, pred_list = run_my_net(mode_name, device, dataloaders, nepocs, nclasses, dataset_sizes)
visualize(mode_name, epoch_losses_train_and_valid, epoch_loss_test, epoch_acc_test, label_list, pred_list)