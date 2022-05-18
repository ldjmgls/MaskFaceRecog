"""
Load face image dataset
"""
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def mask_on_dataloader(dataloader, percentage):
    """ The idea here is to mask a `percentage` of the faces in the loader with a mask
        Here we use `aqeelanwar/MaskTheFace` to create these synthetic images
        
        Returns a new dataloader which contains the masked faces
    """ 
    pass


def create_dataloader(data_dir, batch_size, workers):
    """ The folder is organized as follows: a trainset called `train`, a validation set called `val`
        Each of them have numbered folders
        
        This method recurses through each of the folders to create two dataloaders: one for train and one for validation. 
        Returns two dataloaders, one train, one validation
    """
    pass
