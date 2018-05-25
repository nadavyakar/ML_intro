import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn

lr=0.001
nfc0=100
nfc1=50
nclasses=10
nfc2=nclasses
image_size=28*28

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data', train=True, download=True, transform=transforms), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data', train=False, transform=transforms), batch_size=64, shuffle=True)

class FirstNet(nn.Module):
    def __init__(self,image_size):
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

# optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.AdaDelta(model.parameters(), lr=lr)
# optimizer = optim.RMSprop(model.parameters(), lr=lr)
def train(model,optimizer):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
optimizer
def test(model,):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

optimizer = optim.SGD(model.parameters(), lr=lr)
for model in [FirstNet(image_size=image_size)]:
    for epoch in range(10):
        train(model,optimizer)
        test(model,optimizer)