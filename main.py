import dataloader
import model
import trainer
import torch
import torch.nn as nn

identities = 3702
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 256
workers = 4

train_loader, test_loader, classes = dataloader.fetch_dataloader(batch_size, workers)

net = model.FocusFace(identities = identities)
net.to(device)

trainer.train(net, train_loader, test_loader, n_epochs = 500, lr = 0.01)
