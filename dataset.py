import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.io import decode_image


class DetectionDataset(Dataset):
    """ 
    annotations_file contains the feature-label mapping file
    img_dir contains the directory that is used for storing all the images 
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
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
