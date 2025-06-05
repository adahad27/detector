import torch
import numpy as np
from model import Conv_Net

def train_epoch(data_loader, model, criterion, optimizer):
    for i, (X, y) in enumerate(data_loader):
        """
        We want to zero out the gradients every time we do a backpropagation 
        because PyTorch accumulates gradients when doing backprop.
        """
        optimizer.zero_grad()

        """
        Calling the function like this is like invoking the forward() function,
        except PyTorch calls some extra background operations. We are not meant
        to call the forward() function directly.
        """
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward() # Backpropagation

        optimizer.step() # Gradient descent


def predict_class(logits):
    """
    Assume that we are given the following logits [-1, -2], [1, 3], [-1, 2],
    we should output the predictions [0, 1, 1] because we want to select the 
    highest prediction value out of the results.
    """
    return np.argmax(logits, axis = 1)

def eval_epoch():

    return

def main():


    learning_rate = .001

    model = Conv_Net()

    """
    We use Cross Entropy Loss because the slope of the derivative when we have
    a bad guess is way higher than with something like squared residuals.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters, lr= learning_rate)