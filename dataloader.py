"""
Load face image dataset
"""
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def fetch_dataloader(data_dir, batch_size, workers):
    # TODO
    pass