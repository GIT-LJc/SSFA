"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, batch_size):
        super(ContrastiveLoss,self).__init__()
        self.batch_size = batch_size


    def forward(self,out_1,out_2,temperature=0.5):
        from torch.cuda.amp import autocast as autocast
        with autocast():
            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.batch_size, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * self.batch_size, -1)

            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
