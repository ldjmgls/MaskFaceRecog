"""
Evaluate the model
"""
import argparse
import logging
import os

import numpy as np
import torch

import model
import dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default = "data/faces_emore",
                    help = "Directory containing the dataset")
parser.add_argument("--model_dir", default = "",
                    help = "Directory saving the model and log files")
parser.add_argument("--pretrained", default = None,
                    help = "Optional, filename in --model_dir containing weights to load")  # 'best' or 'train'


def evaluate(net, data_loader, ):
    net.eval()
    with torch.no_grad():
        pass

    logging.info("Validation AUC: {:.3f}".format())
    # print("Validation AUC: {:.3f}".format())