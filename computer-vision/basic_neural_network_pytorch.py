import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#iris dataset
X = torch.tensor(([5.1, 3.5, 1.4], [6.7, 3.1, 4.4], [6.5, 3.2, 5.1]), dtype=torch.float) # 3 X 3 tensor
Y = torch.tensor(([0.2], [1.4], [2]), dtype=torch.float) # 3 X 1 tensor
xInput = torch.tensor(([5.9, 3, 5.1]), dtype=torch.float)

class Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
    
model = Neural_Network()
model.train()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
for i in range(1000):
    optimizer.zero_grad()
    output = model(X)
    loss = F.binary_cross_entropy_with_logits(output, Y)
    print loss
    loss.backward()
    optimizer.step()

print model(xInput)
