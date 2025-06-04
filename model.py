import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()

        #Defining layers here

        self.conv1 = nn.Conv2d(3, 16, 5, 2, 2) # 64 = 1 + floor((64 + 2*2 - 5)/1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 64, 5, 2, 2) #input size becomes 16x16
        self.conv3 = nn.Conv2d(64, 8, 5, 2, 2) #input becomes 4x4
        self.fc_1 = nn.Linear(32, 2)
    
    def init_weights(self):
        torch.manual_seed(42)
        """ 
        Initialize weights in the Convolutional and the Fully connected layer
        according to the normal distribution with mean = 0, and their respective 
        std_dev.s 
        """
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)


        nn.init.normal_(self.fc1.weight, 0.0, 1/ sqrt(self.fc1.weight.size(1)))
        nn.init.constant_(self.fc1.bias, 0.0)



    def forward(self, x):

        #Defining forward pass for data
        conv_block_1 = self.pool(F.relu(self.conv1(x)))
        conv_block_2 = self.pool(F.relu(self.conv2(conv_block_1)))
        conv_block_3 = F.relu(self.conv3(conv_block_2))
        flattened_x = torch.flatten(conv_block_3, 1)
        results = self.fc1(flattened_x)
        return results