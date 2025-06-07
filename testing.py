import torch
from torch import cat
import numpy as np
from model import Conv_Net
from PIL import Image
from sklearn import metrics
from dataset import return_all_datasets
from training import load_model, eval_epoch, predict_class
from torchvision.io import decode_image
from torch.nn.functional import softmax
import os


def test_image(img_path):
    #Create and load the model with it's pretrained parameters here.
    model = Conv_Net()
    load_model(model, "model_parameters.pt")

    #Resize image if needed
    image = Image.open(img_path)
    image = image.resize((64, 64))
    image.save(img_path)

    image = decode_image(img_path)
    image = image.float()
    image = image.resize(1, 3, 64, 64)
    if(image.size(dim = 0) == 1):
        image = cat((image, image, image), dim = 0)
    with torch.no_grad():
        output = model(image)
        probabilites = softmax(output)
        output = predict_class(output)
    return output[0].item(), probabilites[0][output[0].item()].item()



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
    print("Testing has finished!")

if __name__ == "__main__":
    path = input("Please entire the path to the file you want to test:")
    while(not os.path.isfile(path)):
        path = input("Sorry this file doesn't exist, please try again")
    
    prediction, probability = test_image(path)
    print(f"We predict a class of {prediction} with probability {probability}")
    # main()
