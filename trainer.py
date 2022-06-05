"""
Train the model
"""
import argparse
import os
import logging
import numpy as np
from random import sample
from tqdm import tqdm
import wandb

import utils
import dataloader
import model

import torch
import torch.nn as nn
import torch.optim as optim

from eval import evaluate
# from metrics import calculate_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default = "data",
                    help = "Directory containing the dataset")
parser.add_argument("--model_dir", default = "result/base_model",
                    help = "Directory saving the model and log files")
parser.add_argument("--resume", default = False,
                    help = "whether to resume training") 
parser.add_argument("--pretrained", default = None,
                    help = "Optional, filename in --model_dir containing weights to reload before \
                    training")  # "best" or "last"

def train(net, train_loader, val_loader, n_epochs, lr, model_dir, resume = False, pretrained = None):
    """
    Args:
        pretrained: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # Initilization    
    # param = list(net.module.model.parameters()) + list(net.module.fc1.parameters()) + list(net.module.fc2.parameters())
    param = list(net.model.parameters()) + list(net.fc1.parameters()) + list(net.fc2.parameters())
    optimizer = optim.SGD(param, lr = lr, weight_decay = 5e-4, momentum = 0.9)
    criterion = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()
    if pretrained is not None:
        pretrain_path = os.path.join(
            model_dir, pretrained + ".pth.tar")
        logging.info("Loading parameters from {}".format(pretrain_path))
        checkpoint = utils.load_checkpoint(pretrain_path, net, optimizer)
    
    if resume: 
        start_epoch = checkpoint["epoch"]
        best_score = checkpoint["best_score"]
        avg_loss_lst = checkpoint["avg_loss"]
    else: 
        start_epoch = 0
        best_score = 100
        avg_loss_lst = []
    rate_decrease = 1
    patience = 1
    total_step = len(train_loader)

    for epoch in range(start_epoch, n_epochs):
        logging.info("Epoch [{}/{}]".format(epoch + 1, n_epochs))
        net.train()
        total_loss = 0
        # Use tqdm for progress bar
        # with tqdm(total = len(train_loader)) as t_loader:
            # for i, sample in enumerate(train_loader):
        t_loader = tqdm(train_loader)
        for i, sample in enumerate(t_loader, 0):
            # t_loader.set_description("Epoch [{}/{}]".format(epoch + 1, n_epochs))
            input = sample["unmasked"]
            input_masked = sample["masked"]
            label_id = sample["target"]
            label_mask = sample["is_mask"]
            input, input_masked, label_id, label_mask = input.to(device), input_masked.to(device), label_id.to(device), label_mask.to(device)
            
            optimizer.zero_grad()
            # L_unmasked: L_arc + lambda * L_ce
            output, embed1, embed2, mask = net(input, label_id)
            loss = criterion(output, label_id) + 0.1 * criterion(mask * 0, label_mask)      # lambda = 0.1
            
            # L_masked: L_arc + lambda * L_ce
            output_m, embed1_m, embed2_m, mask_m = net(input_masked, label_id)
            loss += criterion(output_m, label_id) + 0.1 * criterion(mask_m, label_mask)
            
            # L_comb = alpha * L_mse * beta * (L_masked + L_unmasked)
            loss /= 2   #
            loss += MSE(embed1, embed1_m) / 3
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
            t_loader.set_description("- Step [{}/{}], Loss: {:.9f}".format(i + 1, total_step, loss.item()))
            # 74907 images, 1085 batches (batch_size = 64), print out loss every 100 batches
            if (i + 1) % 100 == 0:
            # t_loader.set_postfix_str("Step [{}/{}], Loss: {:.5f}".format(i + 1, total_step, loss.item()))
              logging.info("\n- Step [{}/{}], Loss: {:.9f}".format(i + 1, total_step, loss.item()))
        
        # average loss for whole training data: sum(loss per batch) / # of batch
        avg_loss_lst.append(total_loss / total_step)
        logging.info("- Training loss: {:.9f}".format(total_loss / total_step))
        
        # Run for 1 epoch on validation set
        # logging.info("- Start validation ...")
        # metrics = evaluate(model_dir, net, val_loader, device)
        # logging.info("- Validation metrics: {}".format(metrics))
        # last_json_path = os.path.join(model_dir, "val_metrics_last.json")
        # utils.save_dict_to_json(metrics, last_json_path)
        
        is_best = False
        # is_best = metrics["FMR100"] < best_score
        # if metrics["FMR100"] < best_score:
        #     logging.info("- Found new best FMR100: {}".format(metrics["FMR100"]))
        #     best_score = metrics["FMR100"]
        #     patience = 1
        #     best_json_path = os.path.join(model_dir, "val_metrics_best.json")
        #     logging.info("- Found best val metrics: {}".format(metrics))
        #     utils.save_dict_to_json(metrics, best_json_path)
        # else:
        #     if patience == 0:
        #         patience = 1
        #         rate_decrease /= 10
        #         optimizer = optim.SGD(param, lr * rate_decrease, weight_decay = 5e-4, momentum = 0.9)
        #         logging.info("- New Learning Rate: {}".format(lr * rate_decrease))
        #     else: patience -= 1    
        # Save weights
        utils.save_checkpoint({"epoch": epoch + 1, 
                            "state_dict": net.state_dict(),
                            "optim_dict": optimizer.state_dict(),
                            "avg_loss": avg_loss_lst,
                            "best_score": best_score}, 
                            is_best = is_best, 
                            checkpoint = model_dir)      

    utils.plot_trend("train", avg_loss_lst, "Loss", args.model_dir)
    logging.info("Finish training!")


if __name__ == '__main__':
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(230)
    else:
        device = torch.device("cpu")

    # Create output directory for current training if not exists
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, "train.log"))

    # get dataloaders
    batch_size = 64
    workers = 2
    logging.info("Loading the datasets ...")
    train_loader, val_loader = dataloader.create_dataloader(args.data_dir, batch_size, workers)
    logging.info("- Done.")

    # no. of classes (people identities)
    identities = 601
    net = model.FocusFace(identities = identities)
    net.to(device)

    n_epochs = 100
    lr = 0.1
    logging.info("Start training for {} epoch(s) with lr = {} ...".format(n_epochs, lr))
    train(net, train_loader, val_loader, n_epochs, lr, args.model_dir, args.resume, args.pretrained)

