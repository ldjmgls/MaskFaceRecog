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

def create_dataloader(data_path, batch_size, workers):
    """ The folder is organized as follows: a trainset called `train`, a validation set called `val`
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

    train_loader, val_loader = loader(PairDataset(train_path, img_transform)), loader(PairDataset(val_path, img_transform))
    for mimage, uimage, label in train_loader:
        print(mimage, uimage, label)

    return train_loader, val_loader

class PairDataset(ImageFolder):

    def __init__(self, root, transform=None, loader=datasets.folder.default_loader, is_valid_file=None):
        super(PairDataset, self).__init__(root, transform=transform, is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.transform = transform

    def __len__(self):
        return len(self.imgs)//2

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        original_path = path
        try: 
            masked_sample, unmasked_sample = self.transform(self.loader(path)), self.transform(self.loader(path.replace('.jpg', '_N95.jpg')))
            sample = {'masked': masked_sample, 'unmasked': unmasked_sample, 'target' : target}
            sample = (masked_sample, unmasked_sample, target)
        except:
            masked_sample, unmasked_sample = self.transform(self.loader(path.replace('_N95.jpg', '.jpg'))), self.transform(self.loader(path))
            sample = {'masked': masked_sample, 'unmasked': unmasked_sample, 'target' : target}
            sample = (masked_sample, unmasked_sample, target)
        print(sample)
        return sample

     #   if self.transform is not None:
     #       sample = self.transform(sample)
     #   if self.target_transform is not None:
     #       target = self.target_transform(target)

