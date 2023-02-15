import torch
from torch import nn
from torchmetrics.classification import BinaryF1Score

class BinaryF1Loss(nn.Module):
    def __init__(self):
        super(BinaryF1Loss, self).__init__()
        self.score = BinaryF1Score()

    def forward(self, out, batch_y):
        loss = 1 / self.score(out, batch_y) ** 2
        return loss
