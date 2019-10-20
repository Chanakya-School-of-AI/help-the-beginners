# author Vipul 
# `wget https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

# We want to predict/calculate sepal width given the rest three features

# Are any feature engineering required? 
# Ans - No. All rest three things affect the sepal width so I will take them all.
# But I don't know whether I need to normalize stuff or not.

# To-do: Write a data loader of iris dataset 

# How do you write the data loader?
# Pandas -> csv # good option with constraints (does it parallelize data to GPUs?)
# Python file library, read line by line and use numpy to create arrays and store them in memory. # absolute disaster
# Dataloader(torch.data) -> csv good option (it does parallelize data to GPU)

# 150 datapoints

# Neural Network

# 3 data points - [sl pl pw]

class Lecture6(nn.Module):
    def __init__(self):
        super(Lecture6,self).__init__()
        self.fc1 = nn.Linear(3,10)
        self.fc2 = nn.Linear(10,1)
    
    # In tensorflow we specify graphs.
    # In pytorch we don't specify graphs, pytorch creates
    # graphs for us. dynamic graphs!
    def forward(self,x):
        x =  F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
    
neural_net = Lecture6()
loss_function = nn.MSELoss()
optimizer = optim.SGD(neural_net.parameters(), lr=0.01)

# training logic

# inilization of weights? 
# Xavier Intialization -> default

# import the data from dataloader
number_of_epoch = 5

for epoch in range(number_of_epoch):
    # load data and pass it to neural network
    input_arr, ground_truth_value = data
    optimizer.zero_grad()
    
    # forward prop
    ouput = neural_net(input_arr)

    loss = loss_function(ouput,ground_truth_value)

    #backward prop
    loss.backward() # gradients calculated 
    optimizer.step() # take steps

    # How do you calculate various metrics here?
    # or do you calculate metrics during validation or test phase?




