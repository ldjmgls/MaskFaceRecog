"""
End-to-end training
Now, move to trainer so ignore this file temporarily
"""
# import argparse
# import os
# import logging

# import utils
# import dataloader
# import model
# import trainer

# import torch
# import torch.nn as nn

# parser = argparse.ArgumentParser()
# parser.add_argument("--data_dir", default = "data/faces_emore",
#                     help = "Directory containing the dataset")
# parser.add_argument("--model_dir", default = "",
#                     help = "Directory saving the model and log files")
# parser.add_argument("--pretrained", default = None,
#                     help = "Optional, filename in --model_dir containing weights to reload before \
#                     training")  # 'best' or 'train'

# args = parser.parse_args()

# torch.backends.cudnn.benchmark = True
# # Set the random seed for reproducible experiments
# torch.manual_seed(230)
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     torch.cuda.manual_seed(230)
# else:
#     device = torch.device("cpu")

# # Create output directory for current training if not exists
# if not os.path.exists(args.model_dir):
#     os.makedirs(args.model_dir)

# # Set the logger
# utils.set_logger(os.path.join(args.model_dir, "train.log"))

# # get dataloaders
# batch_size = 256
# workers = 4
# logging.info("Loading the datasets ...")
# train_loader, val_loader, classes = dataloader.fetch_dataloader(args.data_dir, batch_size, workers)
# logging.info("- Done.")

# # no. of classes (people identities)
# identities = 1868
# net = model.FocusFace(identities = identities)
# net.to(device)

# n_epochs = 100
# lr = 0.01
# logging.info("Starting training for {} epoch(s) ...".format(n_epochs))
# trainer.train(net, train_loader, val_loader, n_epochs, lr, args.pretrained, device)
