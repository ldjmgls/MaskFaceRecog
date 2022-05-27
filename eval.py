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
from metrics import evaluate_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default = "data/val",
                    help = "Directory containing the dataset")
parser.add_argument("--model_dir", default = "result/base_model",
                    help = "Directory saving the model and log files")
parser.add_argument("--pretrained", default = "last",
                    help = "Optional, filename in --model_dir containing weights to load")  # 'best' or 'last'

def generate_gscore():
    """
    TODO: Genuine match scores are obtained by matching feature sets of the same class (same person)
    """
    pass

def generate_iscore():
    """
    TODO: Impostor matching scores are obtained by matching feature sets of different classes (different persons)
    """
    pass

def evaluate(net, data_loader, ):
    net.eval()
    with torch.no_grad():
        gscores = []
        iscores = []

        for (idx, (x_batch, y_batch)) in enumerate(data_loader):
            y_pred = net(x_batch)
            gscores.append(generate_gscore(y_pred, y_batch))
            iscores.append(generate_iscore(y_pred, y_batch))

    evaluate_metrics(gscores, iscores, clf_name='A', print_results=True)
    # logging.info("- Validation metrics: {}".format(result))


if __name__ == '__main__':
    """
        Evaluate the model on the test set
    """
    args = parser.parse_args()

    torch.manual_seed(230)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(230)
    else:
        device = torch.device("cpu")

    utils.set_logger(os.path.join(args.model_dir, "test.log"))
    batch_size = 64
    workers = 2
    logging.info("Loading the datasets ...")
    _, test_loader = dataloader.create_dataloader(args.data_dir, batch_size, workers)
    logging.info("- Done.")

    identities = 1506   
    net = model.FocusFace(identities).to(device)
    # Load weights from the saved file
    pretrain_path = os.path.join( args.model_dir, args.pretrained + ".pth.tar")
    logging.info("Loading parameters from {}".format(pretrain_path))
    net, optimizer, _ = utils.load_checkpoint(pretrain_path, net)

    logging.info("Start evaluation ...")
    test_metrics = evaluate(net, test_loader, batch_size)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.pretrained))
    utils.save_dict_to_json(test_metrics, save_path)