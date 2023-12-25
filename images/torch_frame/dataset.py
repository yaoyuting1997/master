# 开源数据集
# Code for processing data samples can get messy and hard to maintain; 
# we ideally want our dataset code to be decoupled from our model training code for better readability and modularity. 
# PyTorch provides two data primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset that allow you to use pre-loaded datasets as well as your own data. 
# Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

# PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that subclass torch.utils.data.Dataset and implement functions specific to the particular data. 
# They can be used to prototype and benchmark your model. You can find them here: Image Datasets, Text Datasets, and Audio Datasets

# Loading a Dataset
# Here is an example of how to load the Fashion-MNIST dataset from TorchVision. 
# Fashion-MNIST is a dataset of Zalando’s article images consisting of 60,000 training examples and 10,000 test examples. 
# Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.

# We load the FashionMNIST Dataset with the following parameters:
# root is the path where the train/test data is stored,

# train specifies training or test dataset,
# download=True downloads the data from the internet if it’s not available at root.
# transform and target_transform specify the feature and label transformations
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataset

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)



# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__. Take a look at this implementation; 
# the FashionMNIST images are stored in a directory img_dir, and their labels are stored separately in a CSV file annotations_file.

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        '''
        The __init__ function is run once when instantiating the Dataset object. 
        We initialize the directory containing the images, the annotations file, and both transforms (covered in more detail in the next section).

        The labels.csv file looks like:
        tshirt1.jpg, 0
        tshirt2.jpg, 0
        ......
        ankleboot999.jpg, 9

        '''
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        '''
        The __len__ function returns the number of samples in our dataset.
        '''
        return len(self.img_labels)

    def __getitem__(self, idx):
        '''
        The __getitem__ function loads and returns a sample from the dataset at the given index idx. 
        Based on the index, it identifies the image’s location on disk, converts that to a tensor using read_image, 
        retrieves the corresponding label from the csv data in self.img_labels, calls the transform functions on them (if applicable), 
        and returns the tensor image and corresponding label in a tuple.
        '''
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# The Dataset retrieves our dataset’s features and labels one sample at a time. 
# While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, 
# and use Python’s multiprocessing to speed up data retrieval.

# DataLoader is an iterable that abstracts this complexity for us in an easy API.




