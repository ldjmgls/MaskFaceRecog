"""
Train the model
"""
from collections import defaultdict
import logging
import numpy as np
import os
from random import sample
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim

from metrics import calculate_metrics

def train(net, train_loader, val_loader, n_epochs = 10, lr = 0.1, pretrained = None, device = "cuda"):
    """
    """
    # Initilization
    if pretrained is not None:
        pass
    param = list(net.module.model.parameters()) + list(net.module.fc1.parameters()) + list(net.module.fc2.parameters())
    optimizer = optim.SGD(param, lr = lr, weight_decay = 5e-4, momentum = 0.9)
    criterion = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()

    # best_score = 100
    # rate_decrease=1
    # patience = 1

    for epoch in range(n_epochs):
        logging.info("Epoch {} / {}".format(epoch + 1, n_epochs))
        net.train()

        # Use tqdm for progress bar
        with tqdm(total = len(train_loader)) as t_loader:
            for i, sample in enumerate(t_loader):
                t_loader.set_description("Epoch: {} / {}".format(epoch + 1, n_epochs))
                input = sample["img"]
                input_masked = sample["img_masked"]
                label_id = sample["identity"]
                label_mask = sample["mask"]
                input, input_masked, label_id, label_mask = input.to(device), input_masked.to(device), label_id.to(device), label_mask.to(device)
                
                optimizer.zero_grad()
                # unmasked: L_arc + lambda * L_ce
                output, embed1, embed2, mask = net(input, label_id)
                loss = criterion(output, label_id) + 0.1 * criterion(mask * 0, label_mask)
                # masked: L_arc + lambda * L_ce
                output_m, embed1_m, embed2_m, mask_m = net(input_masked, label_id)
                loss += criterion(output_m, label_id) + 0.1 * criterion(mask_m, label_mask)
                loss /= 2
                loss *= MSE(embed1, embed1_m) / 3
                loss.backward()
                optimizer.step()

                # loss for each batch
                t_loader.set_postfix("Training loss: {:.3f}".format())
        
        # average loss for whole training data <- sum(loss per batch) / # of batch
        # logging.info("- Train metrics: " + metrics_string)
        
        # Run for 1 epoch on validation set
        fmr100 = validate(net, val_loader, )

        logging.info("Finish training!")


def validate(net, val_loader, ):
    net.eval()
    with torch.no_grad():
        pass

    logging.info("Validation AUC: {:.3f}".format())
    # print("Validation AUC: {:.3f}".format())

def test():
    pass


