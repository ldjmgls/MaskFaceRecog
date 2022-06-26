"""
Model Architecture
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import iresnet


class FocusFace(nn.Module):
    def __init__(self, identities=1000):
        super(FocusFace, self).__init__()
        self.model = iresnet.iresnet50()
        self.model.fc = EmbeddingHead(512, 32)
        self.fc1 = ArcFace(512, identities, scale=64, margin=0.5)
        self.fc2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x, label, inference=False):
        embed1, embed2 = self.model(x)              # returns from EmbeddingHead
        embed1 = embed1.view(embed1.shape[0], -1)
        
        if not inference:
            y1 = self.fc1(embed1.view(embed1.shape[0], -1), label)   # ArcFace loss
            y2 = self.fc2(embed2.view(embed2.shape[0], -1))          # Masked?
            embed2 = embed2.view(embed2.shape[0], -1)
            return y1, embed1, embed2, y2
        else:
            y2 = self.fc2(embed2.view(embed2.shape[0], -1))

        return None, embed1, None, F.softmax(y2)[:, 1]


class EmbeddingHead(nn.Module):
    """
    Replace last layer of IResNet with two parallel fully-connected layers

    Args:
        e1: (int) ouput size of 1st embeddings
        e2: (int) ouput size of 2nd embeddings

    Returns:
        1st embeddings: (size of 512) for facial features extracted for the recognition task
        2nd embeddings: (size of 32) for mask detection
    """

    def __init__(self, e1=512, e2=256):
        super(EmbeddingHead, self).__init__()
        self.conv1 = nn.Conv2d(512, e1, kernel_size=(7, 7), stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(512, e2, kernel_size=(7, 7), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(e1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(e2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        size = int(np.sqrt(x.shape[1] / 512))
        x = x.view((x.shape[0], -1, size, size))
        return self.bn1(self.conv1(x)), self.relu(self.bn2(self.conv2(x)))


class ArcFace(nn.Module):
    """
    Implementation of large margin arc distance
        [References]
        - https://github.com/ronghuaiyang/arcface-pytorch.git -> models/metrics.py
        - https://github.com/foamliu/InsightFace-v2.git -> models.py
        - https://github.com/deepinsight/insightface.git -> recognition/arcface_torch/losses.py
        - https://github.com/TreB1eN/InsightFace_Pytorch.git -> model.py

    Args:
        in_features: (int) size of each input sample, embedding size
        out_features: (int) size of each output sample, is equivalent to #classes
        s: (float) norm (scale) of input feature
        m: (float) margin

        cos(theta + m)
    """

    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = scale
        self.m = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = np.cos(margin)
        self.sin_m = np.sin(margin)
        self.theta = np.cos(np.pi - margin)
        self.mm = np.sin(np.pi - margin) * margin

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp((1.0 - torch.pow(cosine, 2)), 1e-9, 1))
        phi = cosine * self.cos_m - sine * self.sin_m           # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.theta, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=label.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -----------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
