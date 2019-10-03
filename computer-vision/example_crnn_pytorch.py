import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # max-pooling and monotonely increasing non-linearities commute. 
        # This means that -
        # MaxPool(Relu(x)) = Relu(MaxPool(x)) 
        # for any input. 
        # So it is technically better to first subsample 
        # through max-pooling and then apply the non-linearity 
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        return x


class ChanakyaCRNN(nn.Module):
    def __init__(self):
        super(ChanakyaCRNN, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(
            input_size=320, 
            hidden_size=64, 
            bidirectional=True)
        self.linear = nn.Linear(64,10)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out2 = self.linear(r_out[:, -1, :])
        
        return F.log_softmax(r_out2, dim=1)