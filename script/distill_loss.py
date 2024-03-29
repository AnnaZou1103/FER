# 2022.10.14-Changed for building manifold kd
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#
# Modified from Fackbook, Deit
# {haozhiwei1, jianyuan.guo}@huawei.com
#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.nn import functional as F


class DistillationLoss(nn.Module):
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.distillation_type = 'soft'  # Distillation type includes soft and hard
        self.tau = 0.2

        self.layer_ids_s = [0, 1, 2, 3, 20, 21, 22, 23]  # The selected layers for the student model
        self.layer_ids_t = [0, 1, 2, 3, 20, 21, 22, 23]  # The selected layers for the teacher model
        self.alpha = 1.0  # Weight for the knowledge distillation loss
        self.beta = 0.1  # Weight for the manifold distillation loss
        self.w_sample = 0.1  # Weight for intra-sample loss
        self.w_patch = 4  # Weight for inter-sample loss
        self.w_rand = 0.2  # Weight for random sample loss
        self.K = 192  # Sample number

    def forward(self, inputs, outputs, labels):
        block_outs_s = outputs[1]
        if isinstance(outputs[0], torch.Tensor):
            outputs = outputs_kd = outputs[0]
        else:
            outputs, outputs_kd = outputs[0]

        base_loss = self.base_criterion(outputs, labels)

        if self.distillation_type == 'none':
            return base_loss

        with torch.no_grad():
            teacher_outputs, block_outs_t = self.teacher_model(inputs)

        if self.distillation_type == 'soft':  # Knowledge distillation loss
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='batchmean',
                log_target=True
            ) * (T * T)
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        loss_base = (1 - self.alpha) * base_loss
        loss_dist = self.alpha * distillation_loss

        # Manifold distillation loss
        loss_mf_sample, loss_mf_patch, loss_mf_rand = mf_loss(block_outs_s, block_outs_t, self.layer_ids_s,
                                                              self.layer_ids_t, self.K, self.w_sample, self.w_patch,
                                                              self.w_rand)
        loss_mf_sample = self.beta * loss_mf_sample
        loss_mf_patch = self.beta * loss_mf_patch
        loss_mf_rand = self.beta * loss_mf_rand
        return loss_base, loss_dist, loss_mf_sample, loss_mf_patch, loss_mf_rand


def mf_loss(block_outs_s, block_outs_t, layer_ids_s, layer_ids_t, K, w_sample, w_patch, w_rand, max_patch_num=0):
    losses = [[], [], []]  # loss_mf_sample, loss_mf_patch, loss_mf_rand
    for id_s, id_t in zip(layer_ids_s, layer_ids_t):
        extra_tk_num = block_outs_s[0].shape[1] - block_outs_t[0].shape[1]
        F_s = block_outs_s[id_s][:, extra_tk_num:, :]  # Remove additional tokens
        F_t = block_outs_t[id_t]
        if max_patch_num > 0:
            F_s = merge(F_s, max_patch_num)
            F_t = merge(F_t, max_patch_num)
        loss_mf_patch, loss_mf_sample, loss_mf_rand = layer_mf_loss(
            F_s, F_t, K)
        losses[0].append(w_sample * loss_mf_sample)
        losses[1].append(w_patch * loss_mf_patch)
        losses[2].append(w_rand * loss_mf_rand)

    loss_mf_sample = sum(losses[0]) / len(losses[0])
    loss_mf_patch = sum(losses[1]) / len(losses[1])
    loss_mf_rand = sum(losses[2]) / len(losses[2])

    return loss_mf_sample, loss_mf_patch, loss_mf_rand


def layer_mf_loss(F_s, F_t, K):
    F_s = F.normalize(F_s, dim=-1)
    F_t = F.normalize(F_t, dim=-1)

    # Manifold loss among different patches (intra-sample)
    M_s = F_s.bmm(F_s.transpose(-1, -2))
    M_t = F_t.bmm(F_t.transpose(-1, -2))

    M_diff = M_t - M_s
    loss_mf_patch = (M_diff * M_diff).mean()

    # Manifold loss among different samples (inter-sample)
    f_s = F_s.permute(1, 0, 2)
    f_t = F_t.permute(1, 0, 2)

    M_s = f_s.bmm(f_s.transpose(-1, -2))
    M_t = f_t.bmm(f_t.transpose(-1, -2))

    M_diff = M_t - M_s
    loss_mf_sample = (M_diff * M_diff).mean()

    # Manifold loss among random sampled patches
    bsz, patch_num, _ = F_s.shape
    sampler = torch.randperm(bsz * patch_num)[:K]

    f_s = F_s.reshape(bsz * patch_num, -1)[sampler]
    f_t = F_t.reshape(bsz * patch_num, -1)[sampler]

    M_s = f_s.mm(f_s.T)
    M_t = f_t.mm(f_t.T)

    M_diff = M_t - M_s
    loss_mf_rand = (M_diff * M_diff).mean()

    return loss_mf_patch, loss_mf_sample, loss_mf_rand


def merge(x, max_patch_num=196):
    B, P, C = x.shape
    if P <= max_patch_num:
        return x
    n = int(P ** (1 / 2))  # Original patch num at each dim
    m = int(max_patch_num ** (1 / 2))  # Target patch num at each dim
    merge_num = n // m  # Merge every (merge_num x merge_num) adjacent patches
    x = x.view(B, m, merge_num, m, merge_num, C)
    merged = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, m * m, -1)
    return merged
