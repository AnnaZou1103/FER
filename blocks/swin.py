import torch.nn as nn


class Swin(nn.Module):
    def __init__(self, swin):
        super().__init__()
        self.swin = swin
        num_ftrs = swin.head.in_features
        self.head = nn.Linear(num_ftrs, 7)

    def forward(self, x):
        feats = self.swin.forward_features(x)
        feats = feats.mean(dim=1)
        x = self.head(feats)
        return feats, x
