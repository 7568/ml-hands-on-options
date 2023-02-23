import torch
from torch import nn
from torchmetrics.classification import BinaryF1Score,BinaryFBetaScore

class BinaryF1Loss(nn.Module):
    def __init__(self):
        super(BinaryF1Loss, self).__init__()
        self.score = BinaryF1Score()

    def forward(self, out, batch_y):
        loss = 1 / self.score(out, batch_y)
        return loss

class BinaryFBetaLoss(nn.Module):
    def __init__(self):
        super(BinaryFBetaLoss, self).__init__()
        self.score = BinaryFBetaScore(beta=2)

    def forward(self, out, batch_y):
        loss = 1 / self.score(out, batch_y)
        return loss
