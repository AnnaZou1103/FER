import torch.nn as nn
from timm.models.swin_transformer_v2 import swinv2_base_window16_256

from blocks.cbam_block import PatchEmbedCBAM, PatchMergingCBAM
from blocks.se_block import BasicLayerSE
from blocks.swin import Swin


def create_model(model_name='base', class_num=7):
    model = swinv2_base_window16_256(pretrained=True)
    if model_name == 'base' or model_name == 'focal':
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, class_num)
    elif model_name == 'cbam':
        model.patch_embed = PatchEmbedCBAM(model.patch_embed)
        for layer in model.layers:
            if type(layer.downsample) is not nn.Identity:
                layer.downsample = PatchMergingCBAM(layer.downsample)

        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, class_num)
    elif model_name == 'se':
        num_layers = len(model.layers)
        for i_layer in range(num_layers):
            layer = model.layers[i_layer]
            model.layers[i_layer] = BasicLayerSE(dim=layer.dim, layer=layer)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, class_num)
    elif model_name == 'center' or model_name == 'supcon':
        model = Swin(swinv2_base_window16_256(pretrained=True))
    return model
