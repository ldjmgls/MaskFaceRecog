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



def evaluate(net, data_loader, batch_size, data_set):
    """
    TODO: This code is retrieved from the original FocusFace Repo. Still needs to be refined and adjusted for our own dataset/dataloader.
    """
    net.eval()
    with torch.no_grad():
        gscores, iscores = [], []
        data_list, issame_list = data_set[0], data_set[1]
        embeddings_list = []

        for i, batch in enumerate(data_loader):
            pass
        
        # Generating embeddings
        for i, data in enumerate(data_list):
            embeddings = generate_embeddings(net, data, batch_size)

            # Normalize
            embeddings_list.append(embeddings)

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
<<<<<<< HEAD
    if args.pretrained is not None:
        pretrain_path = os.path.join(
            args.model_dir, args.pretrained + ".pth.tar")
        logging.info("Loading parameters from {}".format(pretrain_path))
        utils.load_checkpoint(pretrain_path, net)
=======
    pretrain_path = os.path.join( args.model_dir, args.pretrained + ".pth.tar")
    logging.info("Loading parameters from {}".format(pretrain_path))
    utils.load_checkpoint(pretrain_path, net)

    logging.info("Start evaluation ...")
    test_metrics = evaluate(net, test_loader, batch_size)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.pretrained))
    utils.save_dict_to_json(test_metrics, save_path)
>>>>>>> 5197bf20c123d22e3eca1408fdf33b7881a2ed2a
