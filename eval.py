"""
Evaluate the model. (For validation and testing)
"""
import argparse
import logging
import os

import numpy as np
import torch

import utils
import model
import dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default = "data/val",
                    help = "Directory containing the dataset")
parser.add_argument("--model_dir", default = "",
                    help = "Directory saving the model and log files")
parser.add_argument("--pretrained", default = "None",
                    help = "Optional, filename in --model_dir containing weights to load")  # 'best' or 'train'


def evaluate(net, data_loader, ):
    net.eval()
    with torch.no_grad():
        pass

    logging.info("Validation AUC: {:.3f}".format())
    # print("Validation AUC: {:.3f}".format())


if __name__ == '__main__':
    """
        Evaluate the model on the test set
    """
    args = parser.parse_args()

    # Define model
    identities = 1506
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.FocusFace(identities).to(device)
    # Load weights from the saved file
    if args.pretrained is not None:
        pretrain_path = os.path.join(
            args.model_dir, args.pretrained + ".pth.tar")
        logging.info("Loading parameters from {}".format(pretrain_path))
        utils.load_checkpoint(pretrain_path, net)