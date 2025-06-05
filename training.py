import torch
import numpy as np
from model import Conv_Net
from sklearn import metrics
from torch.nn.functional import softmax



class Result:

    def __init__(self,mode : str, epoch : int, accuracy : float, precision : float, f1score : float, recall : float, loss : float):
        self.mode = mode
        self.epoch = epoch
        self.accuracy = accuracy
        self.precision = precision
        self.f1score = f1score
        self.recall = recall
        self.loss = loss



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

def eval_epoch(model, training_loader, validation_loader, testing_loader, criterion, epoch,includes_testing = False):

    def get_stats(loader):
        y_true, y_pred = [], []
        running_loss = []

        for i, (X, y) in enumerate(loader):
            """ 
            Context manager to make sure that extra memory not consumed when
            running functions that normall run with requires_grad = True
            """
            with torch.no_grad():
                y_true.append(y)

                output = model(X)
                y_pred.append(predict_class(output.data))

                running_loss.append(criterion(output, y).item())

        accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        precision = metrics.precision_score(y_true=y_true, y_pred=y_pred)
        f1score = metrics.f1_score(y_true=y_true, y_pred=y_pred)
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred)
        
        loss = np.mean(running_loss)

        return accuracy, precision, f1score, recall, loss
    
    train_accuracy, train_precision, train_f1score, train_recall, train_loss = get_stats(training_loader)
    training_results = Result("training", epoch, train_accuracy, train_precision, train_f1score, train_recall, train_loss)

    val_accuracy, val_precision, val_f1score, val_recall, val_loss = get_stats(validation_loader)
    validation_results = Result("validation", epoch, val_accuracy, val_precision, val_f1score, val_recall, val_loss)

    if(includes_testing):
        test_accuracy, test_precision, test_f1score, test_recall, test_loss = get_stats(testing_loader)
        testing_results = Result("testing", epoch, test_accuracy, test_precision, test_f1score, test_recall, test_loss)   

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