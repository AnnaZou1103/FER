import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpGhost(nn.Module):
    def __init__(self, pretrained_mlp, act_layer=nn.GELU):
        super().__init__()
        self.mlp = pretrained_mlp
        in_features = self.mlp.fc1.in_features
        out_features = in_features
        hidden_features = in_features*3

        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(0.)
        self.cheap_operation2 = nn.Conv1d(in_features, in_features, kernel_size=1, groups=in_features, bias=False)
        self.cheap_operation3 = nn.Conv1d(in_features, in_features, kernel_size=1, groups=in_features, bias=False)

    def forward(self, x):  # x: [B, N, C]
        x1 = self.fc1(x)   # x1: [B, N, C]
        x1 = self.act(x1)

        x2 = self.cheap_operation2(x1.transpose(1,2))  # x2: [B, N, C]
        x2 = x2.transpose(1,2)
        x2 = self.act(x2)

        x3 = self.cheap_operation3(x1.transpose(1, 2))  # x3: [B, N, C]
        x3 = x3.transpose(1, 2)
        x3 = self.act(x3)

        x = torch.cat((x1, x2, x3), dim=2)  # x: [B, N, 3C]
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttentionGhost(nn.Module):
    def __init__(self, pretrained_attn):
        super().__init__()
        self.attn = pretrained_attn

        half_dim = int(0.5*self.attn.dim)
        qkv_bias = False
        self.q = nn.Linear(self.attn.dim, half_dim, bias=qkv_bias)
        self.k = nn.Linear(self.attn.dim, half_dim, bias=qkv_bias)
        self.v = nn.Linear(self.attn.dim, half_dim, bias=qkv_bias)

        self.cheap_operation_q = nn.Conv1d(half_dim, half_dim, kernel_size=1, groups=half_dim, bias=False)
        self.cheap_operation_k = nn.Conv1d(half_dim, half_dim, kernel_size=1, groups=half_dim, bias=False)
        self.cheap_operation_v = nn.Conv1d(half_dim, half_dim, kernel_size=1, groups=half_dim, bias=False)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q1 = self.cheap_operation_q(q.transpose(1,2)).transpose(1,2)
        k1 = self.cheap_operation_k(k.transpose(1,2)).transpose(1,2)
        v1 = self.cheap_operation_v(v.transpose(1,2)).transpose(1,2)

        q = torch.cat((q, q1), dim=2).reshape(B_, N, self.attn.num_heads, C // self.attn.num_heads).permute(0, 2, 1, 3)
        k = torch.cat((k, k1), dim=2).reshape(B_, N, self.attn.num_heads, C // self.attn.num_heads).permute(0, 2, 1, 3)
        v = torch.cat((v, v1), dim=2).reshape(B_, N, self.attn.num_heads, C // self.attn.num_heads).permute(0, 2, 1, 3)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.attn.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.attn.cpb_mlp(self.attn.relative_coords_table).view(-1, self.attn.num_heads)
        relative_position_bias = relative_position_bias_table[self.attn.relative_position_index.view(-1)].view(
            self.attn.window_size[0] * self.attn.window_size[1], self.attn.window_size[0] * self.attn.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
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
        x = self.attn.proj_drop(x)
        return x