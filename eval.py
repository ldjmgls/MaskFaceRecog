"""
Evaluate the model. (For validation and testing)
"""
import argparse
import logging
import os
from typing import Tuple

import numpy as np
import torch
from sklearn.preprocessing import normalize

import dataloader
import model
import utils
from metrics import evaluate_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/val",
                    help="Directory containing the dataset")
parser.add_argument("--model_dir", default="",
                    help="Directory saving the model and log files")
parser.add_argument("--pretrained", default="None",
                    help="Optional, filename in --model_dir containing weights to load")  # 'best' or 'train'


def generate_embeddings(net, data, batch_size):
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

    return embeddings


def normalize_embeddings(embed1: np.ndarray, embed2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize the two embeddings
    :param embed1: the first embedding
    :param embed2: the second embedding
    :return normalized embed1 and embed2
    """
    embed1 = normalize(embed1)
    embeds = normalize(embed1 + embed2)
    
    return embeds[0::2], embeds[1::2]


def embedding_dist(embed1: np.ndarray, embed2: np.ndarray) -> torch.Tensor:
    """
    Calculates the distance between the two embeddings.
    :param embed1: the first embedding
    :param embed2: the second embedding
    :return distance between embed1 and embed2
    """
    return 1 - torch.cdist(torch.from_numpy(embed1).reshape(1, -1), torch.from_numpy(embed2).reshape(1, -1)) / 2


def evaluate(net: model.FocusFace, data_loader: torch.utils.data.DataLoader, batch_size: int):
    """
    TODO: Still intermediate code. Needs to be tested.
    """
    net.eval()
    with torch.no_grad():
        gscores, iscores = [], []

        for i, (gen, imp) in enumerate(data_loader):
            gen_emb1 = generate_embeddings(net, gen['masked'], batch_size)
            gen_emb2 = generate_embeddings(net, gen['unmasked'], batch_size)
            gen_emb1, gen_emb2 = normalize_embeddings(gen_emb1, gen_emb2)

            imp_emb1 = generate_embeddings(net, imp['masked'], batch_size)
            imp_emb2 = generate_embeddings(net, imp['unmasked'], batch_size)
            imp_emb1, imp_emb2 = normalize_embeddings(imp_emb1, imp_emb2)

            gscores.append(embedding_dist(gen_emb1, gen_emb2))
            iscores.append(embedding_dist(imp_emb1, imp_emb2))

    return evaluate_metrics(gscores, iscores, clf_name='A', print_results=True)


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
