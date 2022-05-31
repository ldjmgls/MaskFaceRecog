"""
Load face image dataset
"""
import pdb
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import os
from os.path import exists
from random import random


def create_dataloader(data_path, batch_size, workers):
    """ 
    The folder is organized as follows: a trainset called `train`, a validation set called `val`
    Each of them have numbered folders. There are masked and unmasked images within the same folders.

    This method recurses through each of the folders to create two dataloaders: one for train and one for validation. Each sample in the dataloader is a pair: the unmasked image and the corresponding masked image.
    Returns two dataloaders, one train, one validation
    """
    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "val")

    def loader(ds):
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers)

    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_loader = loader(PairDataset(train_path, img_transform))
    val_loader = loader(ValDataset(val_path, img_transform))

    return train_loader, val_loader


class PairDataset(ImageFolder):
    def __init__(self, root, transform=None, loader=datasets.folder.default_loader, is_valid_file=None):
        super(PairDataset, self).__init__(root, transform=transform, is_valid_file=is_valid_file)
        self.imgs = self.samples  # contains both masked and unmasked samples
        self.transform = transform

    def __len__(self):
        return len(self.imgs) // 2  # divided by 2

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]  # either a masked image or unmasked
        original_path = path

        if path.endswith("_N95.jpg") and exists(path.replace("_N95.jpg", ".jpg")):
            masked_sample, unmasked_sample = self.transform(self.loader(path)), self.transform(self.loader(path.replace('_N95.jpg', '.jpg')))
            sample = {'masked': masked_sample, 'unmasked': unmasked_sample, 'target': target, 'is_mask': 1}
        elif path.endswith(".jpg") and exists(path.replace(".jpg", "_N95.jpg")):
            masked_sample, unmasked_sample = self.transform(self.loader(path.replace('.jpg', '_N95.jpg'))), self.transform(self.loader(path))
            sample = {'masked': masked_sample,'unmasked': unmasked_sample, 'target': target, 'is_mask': 1}
        else:
            # make random idx
            random_idx = int(random() * self.__len__())
            sample = self.__getitem__(random_idx)

        return sample

     #   if self.transform is not None:
     #       sample = self.transform(sample)
     #   if self.target_transform is not None:
     #       target = self.target_transform(target)


class ValDataset(ImageFolder):
    def __init__(self, root, transform=None, loader=datasets.folder.default_loader, is_valid_file=None):
        super(ValDataset, self).__init__(root, transform=transform, is_valid_file=is_valid_file)
        self.imgs = self.samples  # contains both masked and unmasked samples
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.imgs) // 2

    def __getitem__(self, index):
        """
        Gives a tuple where the first element contains pair of image of the same person, and the second contains pair of image of different person
        :params index (int): the index of sample being retrieved
        """
        gen_path, gen_target = self.imgs[index]

        # idx for the imposter image
        im_unmasked_path = None
        
        while im_unmasked_path is None:
            rand_idx = int(random() * self.__len__())   
            if rand_idx == gen_target:  # Is target an int between 0 and self.len?
                rand_idx = self.__len__() - 1 - gen_target  # make sure rand_idx != target
            im_path, im_target = self.imgs[rand_idx]
            im_unmasked_path = im_path.replace('.jpg', '_N95.jpg') if im_path.endswith('_N95.jpg') else im_path
            im_unmasked_path = im_unmasked_path if exists(im_unmasked_path) else None
        
        masked_path = gen_path if gen_path.endswith('_N95.jpg') else gen_path.replace('.jpg', '_N95.jpg')
        gen_unmasked_path = gen_path.replace('_N95.jpg', '.jpg') if gen_path.endswith('_N95.jpg') else gen_path
        

        # Loading data samples to genuine and imposter
        if exists(masked_path) and exists(gen_unmasked_path):
            masked_sample = self.transform(self.loader(masked_path))
            gen_unmasked_sample = self.transform(self.loader(gen_unmasked_path))
            
            genuine = {'masked': masked_sample, 'unmasked': gen_unmasked_sample, 'target': (gen_target, gen_target), 'is_same': 1}
        else:   # If there is no two image of the same person (masked and unmasked)
            print(masked_path, masked_path, gen_unmasked_path, exists(masked_path), exists(gen_unmasked_path))
            genuine = self.__getitem__(int(random()) * self.__len__())[0]    # If no genuine, recursively find random indices until there is genuine data
        
        if exists(masked_path):
            masked_sample = self.transform(self.loader(masked_path))
            im_unmasked_sample = self.transform(self.loader(im_unmasked_path))

            imposter = {'masked': masked_sample, 'unmasked': im_unmasked_sample, 'target': (gen_target, im_target), 'is_same': 0}
        else:
            print(im_path, im_target, exists(im_unmasked_path))
            imposter = self.__getitem__(int(random()) * self.__len__())[1]

        return genuine, imposter


