import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms_
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

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
        return F.log_softmax(self.fc2(x))

class SecondNet(FirstNet):
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x))

class ThirdNet(nn.Module):
    def __init__(self,image_size,nfc0,nfc1,nfc2):
        super(ThirdNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, nfc0)
        self.bn0 = nn.BatchNorm1d(nfc0)
        self.fc1 = nn.Linear(nfc0, nfc1)
        self.bn1 = nn.BatchNorm1d(nfc1)
        self.fc2 = nn.Linear(nfc1, nfc2)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        return F.log_softmax(self.fc2(x))

def train(train_loader,model,optimizer):
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

nfc0=100
nfc1=50
nclasses=10
nfc2=nclasses
image_size=28*28
nepocs=10
valid_ratio=0.2

experiments=[]
# # experiment with models
# # baseline
# lr=0.01
# batch_size=64
# model=(FirstNet(image_size,nfc0,nfc1,nfc2),1)
# optimizer=(optim.SGD(model[0].parameters(), lr=lr),"sgd")
# experiments.append((lr,batch_size,model,optimizer))
#
# lr=0.01
# batch_size=64
# model=(SecondNet(image_size,nfc0,nfc1,nfc2),2)
# optimizer=(optim.SGD(model[0].parameters(), lr=lr),"sgd")
# experiments.append((lr,batch_size,model,optimizer))
#
# lr=0.01
# batch_size=64
# model=(ThirdNet(image_size,nfc0,nfc1,nfc2),3)
# optimizer=(optim.SGD(model[0].parameters(), lr=lr),"sgd")
# experiments.append((lr,batch_size,model,optimizer))
#
# # experiment with data
# remote:
# lr=0.01
# batch_size=64
# model=(FirstNet(image_size,nfc0,nfc1,nfc2),1)
# optimizer=(optim.SGD(model[0].parameters(), lr=lr),"sgd")
# experiments.append((lr,batch_size,model,optimizer))
#
# # experiment with learning rates
# lr=0.001
# batch_size=64
# model=(FirstNet(image_size,nfc0,nfc1,nfc2),1)
# optimizer=(optim.SGD(model[0].parameters(), lr=lr),"sgd")
# experiments.append((lr,batch_size,model,optimizer))
#
# lr=0.05
# batch_size=64
# model=(FirstNet(image_size,nfc0,nfc1,nfc2),1)
# optimizer=(optim.SGD(model[0].parameters(), lr=lr),"sgd")
# experiments.append((lr,batch_size,model,optimizer))
#
# # experiment with batch sizes
# lr=0.01
# batch_size=1
# model=(FirstNet(image_size,nfc0,nfc1,nfc2),1)
# optimizer=(optim.SGD(model[0].parameters(), lr=lr),"sgd")
# experiments.append((lr,batch_size,model,optimizer))
#
# lr=0.01
# batch_size=8
# model=(FirstNet(image_size,nfc0,nfc1,nfc2),1)
# optimizer=(optim.SGD(model[0].parameters(), lr=lr),"sgd")
# experiments.append((lr,batch_size,model,optimizer))
#
# lr=0.01
# batch_size=32
# model=(FirstNet(image_size,nfc0,nfc1,nfc2),1)
# optimizer=(optim.SGD(model[0].parameters(), lr=lr),"sgd")
# experiments.append((lr,batch_size,model,optimizer))
#
# # experiment with optimizers
# lr=0.01
# batch_size=64
# model=(FirstNet(image_size,nfc0,nfc1,nfc2),1)
# optimizer=(optim.Adam(model[0].parameters(), lr=lr), "adam")
# experiments.append((lr,batch_size,model,optimizer))
#
# lr=0.01
# batch_size=64
# model=(FirstNet(image_size,nfc0,nfc1,nfc2),1)
# optimizer=(optim.RMSprop(model[0].parameters(), lr=lr),"rmsprop")
# experiments.append((lr,batch_size,model,optimizer))
#
# chosen setup
lr=0.05
batch_size=32
model=(ThirdNet(image_size,nfc0,nfc1,nfc2),3)
optimizer=(optim.SGD(model[0].parameters(), lr=lr),"sgd")
experiments.append((lr,batch_size,model,optimizer))

transforms = transforms_.Compose([transforms_.ToTensor(), transforms_.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms)

ntrain = len(train_dataset)
indices = list(range(ntrain))
nvalid = int(valid_ratio * len(train_dataset))
ntrain-=nvalid

validation_idx = np.random.choice(indices, size=nvalid, replace=False)
train_idx = list(set(indices) - set(validation_idx))

train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                sampler=validation_sampler)

test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data', train=False, transform=transforms),
                                          batch_size=batch_size, shuffle=False)

for lr,batch_size, (model,i), (optimizer, optimizer_name) in experiments:
    print("model number: {}, optimizer: {}, learning rate: {}, batch size: {}".format(i,optimizer_name,lr,batch_size))
    epocs_list = list(range(nepocs))
    train_loss=0.0
    avg_train_loss_list=[]
    train_acc=0.0
    train_acc_list=[]
    valid_loss=0.0
    avg_valid_loss_list=[]
    valid_acc = 0.0
    valid_acc_list=[]
    for e in epocs_list:
        print("epoc: {}".format(e))
        train_loss, train_acc = train(train_loader, model,optimizer)

        train_loss, train_acc, dummy_list = test_and_validate(train_loader, model, ntrain)
        avg_train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        valid_loss, valid_acc, dummy_list = test_and_validate(validation_loader, model, nvalid)
        avg_valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
    test_avg_loss, test_acc, pred_list = test_and_validate(test_loader, model, None, True)
    print('Training set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(train_loss, train_acc))
    print('Validation set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(valid_loss, valid_acc))
    print('Test set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(test_avg_loss, test_acc))

    plt.plot(epocs_list,avg_train_loss_list,'red',epocs_list,avg_valid_loss_list,'green',linewidth=1, markersize=1)
    plt.xlabel("epocs")
    plt.savefig("loss.model_{}.optimizer_{}.batch_{}.lr_{}.png".format(i,optimizer_name,batch_size,lr))
    plt.clf()
    plt.plot(epocs_list,train_acc_list,'red',epocs_list,valid_acc_list,'green',linewidth=1, markersize=1)
    plt.xlabel("epocs")
    plt.savefig("accuracy.model_{}.optimizer_{}.batch_{}.lr_{}.png".format(i,optimizer_name,batch_size,lr))
    plt.clf()

    with open("test.pred",'w') as f:
        for pred in pred_list:
            f.write("{}\n".format(pred))