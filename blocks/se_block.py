import torch
import torch.nn as nn
import torch.nn.functional as F


#   https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html

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


class WindowAttentionSE(nn.Module):
    def __init__(self, pretrained_attn):
        super(WindowAttentionSE, self).__init__()
        self.attn = pretrained_attn
        self.se_layer = SELayer(self.attn.dim)

    def extra_repr(self) -> str:
        return f'dim={self.attn.dim}, window_size={self.attn.window_size}, ' \
               f'pretrained_window_size={self.attn.pretrained_window_size}, num_heads={self.attn.num_heads}'

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.attn.q_bias is not None:
            qkv_bias = torch.cat(
                (self.attn.q_bias, torch.zeros_like(self.attn.v_bias, requires_grad=False), self.attn.v_bias))
        qkv = F.linear(input=x, weight=self.attn.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.attn.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.attn.cpb_mlp(self.attn.relative_coords_table).view(-1, self.attn.num_heads)
        relative_position_bias = relative_position_bias_table[self.attn.relative_position_index.view(-1)].view(
            self.attn.window_size[0] * self.attn.window_size[1], self.attn.window_size[0] * self.attn.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.attn.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.attn.num_heads, N, N)
            attn = self.attn.softmax(attn)
        else:
            attn = self.attn.softmax(attn)

        attn = self.attn.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.attn.proj(x)
        x = self.se_layer(x)
        x = self.attn.proj_drop(x)
        return x

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.attn.dim * 3 * self.attn.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.attn.num_heads * N * (self.attn.dim // self.attn.num_heads) * N
        #  x = (attn @ v)
        flops += self.attn.num_heads * N * N * (self.attn.dim // self.attn.num_heads)
        # x = self.proj(x)
        flops += N * self.attn.dim * self.attn.dim
        return flops


from timm.models.swin_transformer_v2 import BasicLayer
import torch.utils.checkpoint as checkpoint


class BasicLayerSE(BasicLayer):
    def __init__(
            self, dim, input_resolution, depth, num_heads, window_size,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            norm_layer=nn.LayerNorm, downsample=None, pretrained_window_size=0):
        super().__init__(self, dim, input_resolution, depth, num_heads, window_size,
                         mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                         norm_layer=nn.LayerNorm, downsample=None, pretrained_window_size=0)
        self.se_layer = SELayer(dim)

    def forward(self, x):
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.se_layer(x)
        x = self.downsample(x)
        return x
