import torch
import torch.nn as nn
from timm.models.swin_transformer_v2 import swinv2_base_window16_256

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# train = swinv2_base_window16_256(pretrained=True)
model = torch.load('checkpoints_se/best.pth')

from ptflops import get_model_complexity_info
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
