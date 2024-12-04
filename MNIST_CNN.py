from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms import RandomAffine

# Define DepthwiseSeparableConv
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(1, 8, bias=False),   # Reduced filters to 8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        # Convolution Block 1
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(8, 16, bias=False),  # Reduced filters to 16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        # Transition Block 1
        self.pool1 = nn.MaxPool2d(2, 2)
        # Convolution Block 2
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(16, 32, bias=False), # Reduced filters to 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        # Convolution Block 3
        self.conv4 = nn.Sequential(
            DepthwiseSeparableConv(32, 32, bias=False), # Kept filters at 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        # Transition Block 2
        self.pool2 = nn.MaxPool2d(2, 2)
        # Convolution Block 4
        self.conv5 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, bias=False), # Reduced filters to 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        # Global Average Pooling and Output
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10, bias=False)        # Adjusted input features to 64

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = x.view(-1, 64)                             # Adjusted for 64 channels
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

torch.manual_seed(1)
batch_size = 128

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Updated data augmentation with RandomAffine
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.RandomAffine(
                           degrees=5,            # Rotate by ±5 degrees
                           translate=(0.1, 0.1), # Translate by ±10%
                           scale=(0.9, 1.1)      # Scale between 90% to 110%
                       ),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))

from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_description(desc=f'loss={loss.item():.4f} batch_id={batch_idx}')


# Modify test function to save the best model
def test(model, device, test_loader):
    global best_test_loss
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print('Best model saved with test loss: {:.4f}'.format(best_test_loss))
    return test_loss  

# Initialize best_test_loss
best_test_loss = float('inf')

model = Net().to(device)
# Adjusted optimizer and scheduler with higher max_lr
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

scheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.5,                      
    steps_per_epoch=len(train_loader),
    epochs=7,                        
    div_factor=5.0,                  
    final_div_factor=1e4             
)

# Update the training loop for 7 epochs
for epoch in range(1, 8):            # Train for 7 epochs
    print(f'\nEpoch {epoch}:')
    train(model, device, train_loader, optimizer, epoch)
    test_loss = test(model, device, test_loader)