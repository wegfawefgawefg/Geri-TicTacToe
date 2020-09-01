import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class Network(torch.nn.Module):
    def __init__(self, alpha, inputChannels, numActions):
        super().__init__()
        self.inputChannels = inputChannels
        self.numActions = numActions
        self.convOutShape = 32 * 9 * 9
        self.fc1Dims = 1024
        self.fc2Dims = 512

        self.conv1 = nn.Conv2d(self.inputChannels, 16, 3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(32)

        # self.maxPool1 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.convOutShape, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)

        #   tail networks
        self.policy = nn.Linear(self.fc2Dims, numActions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        batchSize = x.shape[0]
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = F.relu(x)
        # x = self.maxPool1(x)
        # print("post conv maxpool shape: {}".format(x.shape))

        x = x.view(batchSize, self.convOutShape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy = self.policy(x)
        return policy

if __name__ == "__main__":
    inputShape = (1, 3, 3, 3)

    net = Network(
        alpha=0.001, 
        inputChannels=3, 
        numActions=9)

    #   make fake data
    x = torch.ones(inputShape).to(net.device)
    print(x.shape)

    #   feedforward 
    policy = net(x)
    print("policy {}".format(policy))
