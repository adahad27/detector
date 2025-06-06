import torch
import numpy as np
from model import Conv_Net
from sklearn import metrics
from dataset import return_all_datasets
from training import load_model, eval_epoch
import os

def main():


    learning_rate = .001

    model = Conv_Net()

    """
    We use Cross Entropy Loss because the slope of the derivative when we have
    a bad guess is way higher than with something like squared residuals.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

    patience = 5
    current_patience = 0
    stats = []
    batch_size = 1000

    epoch = 0

    paramter_save_path = "model_parameters.pt"

    if(os.path.isfile(paramter_save_path)):
        load_model(model, paramter_save_path)

    training_loader, validation_loader, testing_loader = return_all_datasets(batch_size=batch_size)

    global_min_loss = float("inf")
    print("Model evaluation for testing has started...")
    eval_epoch(model, 
               training_loader=training_loader, 
               validation_loader=validation_loader, 
               testing_loader=testing_loader, 
               criterion=criterion, 
               epoch = epoch, 
               stats = stats, 
               includes_testing=True)
    stats[-1].print_stats()
    print("Training has finished!")

main()
