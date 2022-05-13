"""
Train the model
"""
from collections import defaultdict
import numpy as np
import os
from tqdm import tqdm
import wandb

import torch
from torch import nn

from metrics import calculate_metrics

def train(net, train_loader, test_loader, n_epochs = 10, lr = 0.1, pretrained = None):
    if pretrained is not None:
        pass


def validate():
    pass

def test():
    pass


