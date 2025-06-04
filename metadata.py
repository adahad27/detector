import os
import numpy as np
import pandas as pd
import random

def assign_datatypes(path):
    random.seed(131)
    df = pd.read_csv(path)
    partitions = []
    for index in range(df.shape[0]):
        partition_number = random.randint(1, 5)
        if(partition_number < 4):
            #Then this is training data
            partitions.append("training")
        elif(partition_number == 4):
            #Then this is validation data
            partitions.append("validation")
        else:
            #Then this must be testing data
            partitions.append("testing")
    df.insert(3, "partition", partitions)
    df = df[["file_name", "label", "partition"]]
    df.to_csv("modified_train.csv")

assign_datatypes("train.csv")