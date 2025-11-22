import torch
import torch.nn as nn
from .triplet_loss import TripletLoss

class Model(nn.Module):

    def __init__(self, hidden_dim, margin, batch_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        from model.resnet import resnet18  # ejemplo
        self.encoder = resnet18(out_dim=hidden_dim)

        self.triplet_loss = TripletLoss(batch_size, margin)

    def forward(self, x):
        emb, aux = self.encoder(x)
        return emb
