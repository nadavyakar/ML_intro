import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class FirstNet(nn.Module):
    def __init__(self,image_size,nfc0,nfc1,nfc2):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, nfc0)
        self.fc1 = nn.Linear(nfc0, nfc1)
        self.fc2 = nn.Linear(nfc1, nfc2)
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

# def validate(train_loader,model):

def measure(output, dataset, target):
    loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(dataset)
    acc = correct / len(dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(dataset), 100. * acc))
    return loss, acc

def train(train_loader,model,optimizer):
    model.train()
    loss_sum=0
    correct = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

        loss = F.nll_loss(output, labels)
        loss_sum += loss.data[0]  # sum up batch loss
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

        loss.backward()
        optimizer.step()
    return loss_sum /len(train_loader.dataset), correct / len(train_loader.dataset)


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * acc))

lr=0.01
nfc0=100
nfc1=50
nclasses=10
nfc2=nclasses
image_size=28*28
nepocs=10

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data', train=True, download=True, transform=transforms), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data', train=False, transform=transforms), batch_size=64, shuffle=True)

i=0
for model in [FirstNet(image_size,nfc0,nfc1,nfc2)]:
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.AdaDelta(model.parameters(), lr=lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)
    i+=1
    epocs_list = list(range(nepocs))
    avg_train_loss_list=[]
    train_acc_list=[]
    avg_valid_loss_list=[]
    valid_acc_list=[]
    for e in epocs_list:
        print("epoc {}".format(e))
        avg_loss,acc = train(train_loader, model,optimizer)
        avg_train_loss_list.append(avg_loss)
        train_acc_list.append(acc)
        # avg_loss, acc = valid(train_loader, model, optimizer)
        # avg_valid_loss_list.append(avg_loss)
        # valid_acc_list.append(acc)
    test(model, test_loader)
    plt.plot(epocs_list,avg_train_loss_list,'bs',epocs_list,train_acc_list,'rs',linewidth=1, markersize=1)
    # plt.plot(epocs_list,avg_train_loss_list,'bs',epocs_list,train_acc_list,'rs',epocs_list,avg_valid_loss_list,'gs',epocs_list,valid_acc_list,'ps')
    plt.xlabel("epocs")
    plt.savefig("performance_model_{}.png".format(1))
    plt.clf()