import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn 
import torch.optim as optim 

transformation = transforms.Compose([transforms.ToTensor(), 
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10("./data/", train=True, transform=transformation, download=True)

# Dataloader
traindataloader = torch.utils.data.DataLoader(trainset, batch_size=4, num_workers=2, shuffle=True)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class CNN(nn.Module):
        def __init__(self):
                super(CNN, self).__init__()
                self.cnn_layer1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
                self.cnn_layer2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
                self.pool = nn.MaxPool2d(2,2)
                self.fc1 = nn.Linear(16*5*5, 120)
                self.fc2 = nn.Linear(120,10)

        def forward(self, x):
                x = self.pool(nn.ReLU(self.cnn_layer1(x)))
                x = self.pool(nn.ReLU(self.cnn_layer2(x)))
                x = x.view(-1, 16*5*5)
                x = nn.ReLU(self.fc1(x))
                x = self.fc2(x)
                return x

net = CNN()

optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
loss_function = nn.CrossEntropyLoss()

epoch = 2
for i in range(epoch):
        running_loss = 0.0
        for i, data in enumerate(traindataloader, 0):
                inputs, labels = data

                optimizer.zero_grad()

                #forward + backprop
                output = net(inputs)
                loss = loss_function(output, label)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                                (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

print("Finished Training")

