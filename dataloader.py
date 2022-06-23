"""
Load face image dataset
"""
import glob
import os
import re
from os.path import basename, dirname, exists
from random import random

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from numpy.random import choice
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def create_dataloader(data_path: str, batch_size: int, workers: int) -> tuple:
    """ 
    The folder is organized as follows: a trainset called `train`, a validation set called `val`
    Each of them have numbered folders. There are masked and unmasked images within the same folders.

    This method recurses through each of the folders to create two dataloaders: one for train and one for validation. Each sample in the dataloader is a pair: the unmasked image and the corresponding masked image.

    :params data_path: the path to the data directory
    :params batch_size: the size of the batch
    :params workers: the number of workers for the dataloader
    :return tuple(train dataloader, validation dataloader)
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
        super(PairDataset, self).__init__(
            root, transform=transform, is_valid_file=is_valid_file)
        self.imgs = self.samples  # contains both masked and unmasked samples
        self.transform = transform

    def __len__(self):
        return len(self.imgs) // 2  # divided by 2

    def __getitem__(self, index: int) -> dict:
        """
        :params index: the index of sample being retrieved
        :return a dict containing a pair of masked and unmasked samples of the same target class
        """
        path, target = self.samples[index]  # either a masked image or unmasked

        if path.endswith("_N95.jpg") and exists(path.replace("_N95.jpg", ".jpg")):
            masked_sample, unmasked_sample = self.transform(self.loader(
                path)), self.transform(self.loader(path.replace('_N95.jpg', '.jpg')))
            sample = {'masked': masked_sample,
                      'unmasked': unmasked_sample, 'target': target, 'is_mask': 1}
        elif path.endswith(".jpg") and exists(path.replace(".jpg", "_N95.jpg")):
            masked_sample, unmasked_sample = self.transform(self.loader(
                path.replace('.jpg', '_N95.jpg'))), self.transform(self.loader(path))
            sample = {'masked': masked_sample,
                      'unmasked': unmasked_sample, 'target': target, 'is_mask': 1}
        else:
            # Get random index when target could not be found
            random_idx = int(random() * self.__len__())
            sample = self.__getitem__(random_idx)

        return sample


class ValDataset(ImageFolder):
    def __init__(self, root, transform=None, loader=datasets.folder.default_loader, is_valid_file=None):
        super(ValDataset, self).__init__(
            root, transform=transform, is_valid_file=is_valid_file)
        self.imgs = self.samples  # contains both masked and unmasked samples
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.imgs) // 2

    def __getitem__(self, index: int) -> dict:
        """
        Gives a tuple where the first element contains pair of image of the same person, and the second contains pair of image of different people
        :params index: the index of sample being retrieved
        """
        gen_path, gen_target = self.imgs[index]

        # idx for the imposter image
        im_unmasked_path = None

        while im_unmasked_path is None:
            rand_idx = int(random() * self.__len__())
            if rand_idx == gen_target:  # Is target an int between 0 and self.len?
                rand_idx = self.__len__() - 1 - gen_target  # make sure rand_idx != target
            im_path, im_target = self.imgs[rand_idx]
            im_unmasked_path = im_path.replace(
                '.jpg', '_N95.jpg') if im_path.endswith('_N95.jpg') else im_path
            im_unmasked_path = im_unmasked_path if exists(
                im_unmasked_path) else None

        # The masked image path
        masked_path = gen_path if gen_path.endswith(
            '_N95.jpg') else gen_path.replace('.jpg', '_N95.jpg')

        # Generating the unmasked image path
        class_folder = dirname(gen_path)
        sample_filename = basename(gen_path)

        sample_num = sample_filename[:-7] if sample_filename.endswith(
            'N95.jpg') else sample_filename[:-4]
        pool = list(filter(re.compile(
            '.+[0-9]+\_N95\.jpg').match, glob.glob(f'{class_folder}/*.jpg')))
        all_sample_nums = [
            file_name for file_name in pool if file_name.endswith('N95.jpg')]

        # Cases when there is only one sample in the directory of a person
        try:
            if len(all_sample_nums) <= 2:
                all_sample_nums.remove(sample_num)
        except:
            pass

        gen_unmasked_path = choice(all_sample_nums)

        # Loading data samples to genuine dataset
        if exists(masked_path) and exists(gen_unmasked_path):
            masked_sample = self.transform(self.loader(masked_path))
            gen_unmasked_sample = self.transform(
                self.loader(gen_unmasked_path))

            genuine = {'masked': masked_sample, 'unmasked': gen_unmasked_sample, 'target': (
                gen_target, gen_target), 'is_same': 1}
        # If there is no two image of the same person (masked and unmasked)
        else:
            print(masked_path, masked_path, gen_unmasked_path,
                  exists(masked_path), exists(gen_unmasked_path))
            # If no genuine, recursively find random indices until there is genuine data
            genuine = self.__getitem__(int(random()) * self.__len__())[0]

        # Loading data samples to imposter dataset
        if exists(masked_path):
            masked_sample = self.transform(self.loader(masked_path))
            im_unmasked_sample = self.transform(self.loader(im_unmasked_path))

            imposter = {'masked': masked_sample, 'unmasked': im_unmasked_sample, 'target': (
                gen_target, im_target), 'is_same': 0}
        else:
            print(im_path, im_target, exists(im_unmasked_path))
            imposter = self.__getitem__(int(random()) * self.__len__())[1]

        return genuine, imposter
