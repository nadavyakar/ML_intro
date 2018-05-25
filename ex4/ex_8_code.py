import numpy as np
import torch
from torchvision import datasets, transforms
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

def test_and_validate(dataset_loader, model, ndataset=None):
    model.eval()
    loss = 0.
    correct = 0.
    if ndataset is None:
        ndataset = len(dataset_loader.dataset)
    for data, target in dataset_loader:
        output = model(data)
        loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    return loss/ ndataset, 100. * correct / ndataset

# lr=0.01
nfc0=100
nfc1=50
nclasses=10
nfc2=nclasses
image_size=28*28
nepocs=10
# batch_size=64
valid_ratio=0.2

for lr in [0.001,0.01, 0.05]:
    print('learning rate'.format(lr))
    for batch_size in [pow(2,x) for x in [0,3,5,6]]:
        print('batch size'.format(batch_size))
        transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset=datasets.FashionMNIST('./data', train=True, download=True, transform=transforms)

        ntrain = len(train_dataset)
        indices = list(range(ntrain))
        nvalid = int(valid_ratio*len(train_dataset))

        validation_idx = np.random.choice(indices, size=nvalid, replace=False)
        train_idx = list(set(indices) - set(validation_idx))

        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(validation_idx)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=validation_sampler)

        test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data', train=False, transform=transforms), batch_size=batch_size, shuffle=True)

        i=0
        for model in [FirstNet(image_size,nfc0,nfc1,nfc2),SecondNet(image_size,nfc0,nfc1,nfc2),ThirdNet(image_size,nfc0,nfc1,nfc2)]:
            i+=1
            print("\nmodel number {}".format(i))
            for optimizer, optimizer_name in [ (optim.SGD(model.parameters(), lr=lr),"sgd"),
                                               (optim.Adam(model.parameters(), lr=lr), "adam"),
                                               (optim.RMSprop(model.parameters(), lr=lr),"rmsprop") ]:
                print("optimizer {}".format(optimizer_name))
                # optimizer = optim.SGD(model.parameters(), lr=lr)
                # optimizer = optim.Adam(model.parameters(), lr=lr)
                # optimizer = optim.AdaDelta(model.parameters(), lr=lr)
                # optimizer = optim.RMSprop(model.parameters(), lr=lr)
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
                    print("epoc {}".format(e))
                    train_loss, train_acc = train(train_loader, model,optimizer)

                    train_loss, train_acc = test_and_validate(train_loader, model, ntrain)
                    avg_train_loss_list.append(train_loss)
                    train_acc_list.append(train_acc)

                    valid_loss, valid_acc = test_and_validate(validation_loader, model, nvalid)
                    avg_valid_loss_list.append(valid_loss)
                    valid_acc_list.append(valid_acc)
                test_avg_loss, test_acc = test_and_validate(test_loader, model)
                print('Training set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(train_loss, train_acc))
                print('Validation set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(valid_loss, valid_acc))
                print('Test set: Average loss: {:.4f}, Accuracy: {:.0f}%'.format(test_avg_loss, test_acc))

                # plt.plot(epocs_list,avg_train_loss_list,color='blue',linewidth=1, markersize=1)
                plt.plot(epocs_list,avg_train_loss_list,'red',epocs_list,avg_valid_loss_list,'green',linewidth=1, markersize=1)
                plt.xlabel("epocs")
                plt.savefig("loss_model_{}.optimizer_{}.batch_{}.lr_{}.png".format(i,optimizer_name,batch_size,lr))
                plt.clf()
                # plt.plot(epocs_list,train_acc_list,color='blue',linewidth=1, markersize=1)
                plt.plot(epocs_list,train_acc_list,'red',epocs_list,valid_acc_list,'green',linewidth=1, markersize=1)
                plt.xlabel("epocs")
                plt.savefig("accuracy_model_{}.optimizer_{}.batch_{}.lr_{}.png".format(i,optimizer_name,batch_size,lr))
                plt.clf()