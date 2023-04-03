import torch
from ptflops import get_model_complexity_info

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load('../checkpoints/small/best.pth')

with torch.cuda.device(0):
    macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
