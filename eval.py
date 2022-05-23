"""
Evaluate the model. (For validation and testing)
"""
import argparse
import logging
import os

import numpy as np
import torch
from sklearn.preprocessing import normalize

import utils
import model
import dataloader
from metrics import evaluate_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/val",
                    help="Directory containing the dataset")
parser.add_argument("--model_dir", default="",
                    help="Directory saving the model and log files")
parser.add_argument("--pretrained", default="None",
                    help="Optional, filename in --model_dir containing weights to load")  # 'best' or 'train'


def generate_scores():
    """
    TODO: Genuine match scores are obtained by matching feature sets of the same class (same person)
    """
    pass


def generate_iscore():
    """
    TODO: Impostor matching scores are obtained by matching feature sets of different classes (different persons)
    """
    pass


def evaluate(net, data_loader, batch_size, data_set):
    """
    TODO: This code is retrieved from the original FocusFace Repo. Still needs to be refined and adjusted for our own dataset/dataloader.
    """
    net.eval()
    with torch.no_grad():
        gscores = []
        iscores = []
        data_list, issame_list = data_set[0], data_set[1]
        embeddings_list = []


        # Generating embeddings
        for i, data in enumerate(data_list):
            embeddings = None
            start = 0

            while start < data.shape[0]:
                end = min(start + batch_size, data.shape[0])
                count = end - start
                _data = data[end - batch_size: end]
                img = (((_data / 255) - 0.5) / 0.5).to(device)
                y_pred = net(img, inference=True)[1]
                _embeddings = y_pred.detach().cpu().numpy()

                if _embeddings is None:
                    embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))

                embeddings[start:end, :] = _embeddings[(batch_size - count):, :]
                start = end

            embeddings_list.append(embeddings)

        # Normalize
        embeddings = embeddings_list[0].copy()
        embeddings = normalize(embeddings)

        embeddings = embeddings_list[0] + embeddings_list[1]
        embeddings = normalize(embeddings)

        embeddings1 = embeddings[0::2]
        embeddings2 = embeddings[1::2]

        # Adding to gscores and iscores
        for embedding1, embedding2, label in zip(embeddings1, embeddings2, issame_list):
            dist = 1 - torch.cdist(torch.from_numpy(embedding1).view(1, -1),
                                   torch.from_numpy(embedding2).view(1, -1))/2
            if label == 1:
                gscores.append(dist)
            else:
                iscores.append(dist)

    evaluate_metrics(gscores, iscores, clf_name='A', print_results=True)
    # logging.info("Validation AUC: {:.3f}".format())
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
