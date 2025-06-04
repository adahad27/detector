import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()

        #Defining layers here

        #Temporary layers created as placeholders within the scaffolding.
        self.conv1 = nn.Conv2d()

        self.hidden1 = nn.Linear(2, 3)
        self.hidden2 = nn.Linear(3, 2)
        self.output = nn.Linear(2, 2)
    
    def forward(self, x):

        #Defining forward pass for data
        z2 = self.hidden1(x)
        h2 = F.sigmoid(z2)
        z3 = self.hidden2(h2)
        h3 = F.sigmoid(z3)
        z4 = self.output(h3)

        return z4