import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from torch import cat
from PIL import Image
from torchvision.io import decode_image
from torchvision.transforms import v2

class DetectionDataset(Dataset):
    """ 
    annotations_file contains the feature-label mapping file
    img_dir contains the directory that is used for storing all the images
    partition contains which set it belongs to out of "training", "testing", and "validation"
    """
    def __init__(self, annotations_file, img_dir, partition, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.partition = partition
        self.target_transform = target_transform
        self.img_labels = self.img_labels[(self.img_labels.partition == self.partition)]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 1]
        
        image = decode_image(img_path)
        image = image.float()
        label = self.img_labels.iloc[idx, 2]
        if(image.size(dim=0) == 1):
            image = cat((image, image, image), dim = 0)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

def return_all_datasets(batch_size):
    training_dataset = DetectionDataset("modified_train.csv", "train_data/", "training")
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    validation_dataset = DetectionDataset("modified_train.csv", "train_data/", "validation")
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    
    testing_dataset = DetectionDataset("modified_train.csv", "train_data/", "testing")
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True)
    
    return training_loader, validation_loader, testing_loader 

class ImageStandardizer():
    """ 
    This class is responsible for normalizing images.
    fit() calculates mean and std. dev. for the image.
    transform modifies the image to fit with the mean and std. dev.
    """

    def __init__(self):
        self.mean = None
        self.std_dev = None
        self.resized = False
    
    def fit(self, X):
        self.mean = np.mean(X, axis = (0, 1, 2))
        self.std_dev = np.std(X, axis = (0, 1, 2))

    def transform(self, X):
        return (X - self.mean)/self.std_dev

def resize(img_dir):
    """ 
    Run this function on the set of images that you have to standardize their
    resolution, and also to make sure that the folder of training data is not
    too large. 
    """
    for file in os.listdir(img_dir):
        if(file[-4:] == ".jpg"):
            img = Image.open(f"{img_dir}/{file}")
            img = img.resize((64, 64))
            img.save(f"{img_dir}/{file}")
