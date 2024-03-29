import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.swin_transformer_v2 import swinv2_base_window16_256, swinv2_small_window16_256


# CBAM module
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=8, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class PatchMergingCBAM(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, downsample):
        super().__init__()
        self.downsample = downsample
        self.cbam_layer = CBAM(downsample.dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.downsample.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        x0 = self.cbam_layer(x0.permute(0, 3, 1, 2))
        x1 = self.cbam_layer(x1.permute(0, 3, 1, 2))
        x2 = self.cbam_layer(x2.permute(0, 3, 1, 2))
        x3 = self.cbam_layer(x3.permute(0, 3, 1, 2))

        x = torch.cat([x0.permute(0, 2, 3, 1), x1.permute(0, 2, 3, 1), x2.permute(0, 2, 3, 1), x3.permute(0, 2, 3, 1)],
                      -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.downsample.reduction(x)
        x = self.downsample.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.downsample.input_resolution}, dim={self.downsample.dim}"

    def flops(self):
        H, W = self.downsample.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.downsample.dim * 2 * self.downsample.dim
        flops += H * W * self.downsample.dim // 2
        return flops


class PatchEmbedCBAM(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_embed):
        super().__init__()
        self.patch_embed = patch_embed
        self.cbam_layer = CBAM(96)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.patch_embed.img_size[0] and W == self.patch_embed.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.patch_embed.img_size[0]}*{self.patch_embed.img_size[1]})."
        x = self.patch_embed.proj(x)
        x = self.cbam_layer(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.patch_embed.norm is not None:
            x = self.patch_embed.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patch_embed.patches_resolution
        flops = Ho * Wo * self.patch_embed.embed_dim * self.patch_embed.in_chans * (
                self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1])
        if self.patch_embed.norm is not None:
            flops += Ho * Wo * self.patch_embed.embed_dim
        return flops


# Squeeze and Excitation module
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x


class BasicLayerSE(nn.Module):
    def __init__(self, dim, layer):
        super(BasicLayerSE, self).__init__()
        self.layer = layer
        self.se_layer = SELayer(dim)

    def forward(self, x):
        for blk in self.layer.blocks:
            if self.layer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.se_layer(x)
        x = self.layer.downsample(x)
        return x


# SwinV2 for contrastive learning
class CustomizedSwin(nn.Module):
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


# SwinV2 for knowledge distillation
class DistilledSwin(nn.Module):
    def __init__(self, swin):
        super().__init__()
        self.swin = swin
        num_ftrs = swin.head.in_features
        self.head = nn.Linear(num_ftrs, 7)

    def forward_features(self, x):
        x = self.swin.patch_embed(x)
        if self.swin.absolute_pos_embed is not None:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.pos_drop(x)

        block_outs = []
        for layer in self.swin.layers:
            x_ = x
            for blk in layer.blocks:
                x_ = blk(x_)
                block_outs.append(x_)
            x = layer(x)

        x = self.swin.norm(x)  # B L C
        return x, block_outs

    def forward(self, x):
        x, block_outs = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x, block_outs


def create_model(model_name='base', class_num=7):
    if model_name == 'distill_small':
        return DistilledSwin(swinv2_small_window16_256(pretrained=True))
    elif model_name == 'distill_base':
        return DistilledSwin(swinv2_base_window16_256(pretrained=True))
    elif model_name == 'center':
        return CustomizedSwin(swinv2_small_window16_256(pretrained=True))

    model = swinv2_small_window16_256(pretrained=True)
    if model_name == 'cbam':
        model.patch_embed = PatchEmbedCBAM(model.patch_embed)
        for layer in model.layers:
            if type(layer.downsample) is not nn.Identity:
                layer.downsample = PatchMergingCBAM(layer.downsample)
    elif model_name == 'se':
        num_layers = len(model.layers)
        for i_layer in range(num_layers):
            layer = model.layers[i_layer]
            model.layers[i_layer] = BasicLayerSE(dim=layer.dim, layer=layer)

    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, class_num)
    return model
