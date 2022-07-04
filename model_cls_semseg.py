#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 13:43:22 2022

@author: linxi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """
    Input:
        points: input points data, [B, C, N]
    Return:
        idx: sample index data, [B, N, K]
    """
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]            # (batch_size, num_points, k)
    return idx


def index_points_neighbors(x, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    batch_size = x.size(0)
    num_points = x.size(1)
    num_dims = x.size(2)

    device = idx.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    neighbors = x.view(batch_size*num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, -1, num_dims)

    return neighbors


def get_neighbors(x, k=20):
    """
    Input:
        points: input points data, [B, C, N]
    Return:
        feature_points:, indexed points data, [B, 2*C, N, K]
    """
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    idx = knn(x, k)                                         # batch_size x num_points x 20
    x = x.transpose(2, 1).contiguous()
    neighbors = index_points_neighbors(x, idx)  
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) 
    feature = torch.cat((neighbors - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN_cls_semseg(nn.Module):
    def __init__(self, args):
        super(DGCNN_cls_semseg, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn6, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn7, 
                                   nn.LeakyReLU(negative_slope=0.2)) 
        self.conv10 = nn.Conv1d(256, 7, kernel_size=1, bias=False)

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, 5)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.float()

        x = get_neighbors(x, k=self.k)
        x1 = self.conv1(x)
        y = self.conv6(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]
        y1 = y.max(dim=-1, keepdim=False)[0]

        x = get_neighbors(x1, k=self.k)
        x2 = self.conv2(x)
        x2 = x2.max(dim=-1, keepdim=False)[0]
        y = get_neighbors(y1, k=self.k)
        y2 = self.conv2(y)
        y2 = self.conv6(y2)
        y2 = y2.max(dim=-1, keepdim=False)[0]

        x = get_neighbors(x2, k=self.k)
        x3 = self.conv3(x)
        x3 = x3.max(dim=-1, keepdim=False)[0]
        y = get_neighbors(y2, k=self.k)
        y3 = self.conv2(y)
        y3 = y3.max(dim=-1, keepdim=False)[0]

        x = get_neighbors(x3, k=self.k)
        x4 = self.conv4(x)
        x4 = x4.max(dim=-1, keepdim=False)[0]

        xs1 = torch.cat((x1, x2, x3, x4), dim=1)
        ys1 = torch.cat((y1, y2, y3), dim=1)
        y = self.conv7(ys1)
        y4 = y.max(dim=-1, keepdim=True)[0]
        y4 = y4.repeat(1, 1, num_points)
        ys2 = torch.cat((y4, y1, y2, y3), dim=1)

        x5 = self.conv5(xs1) 
        xm = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        xa = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)
        xs2 = torch.cat((xm, xa), 1)

        x = F.leaky_relu(self.bn6(self.linear1(xs2)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.linear3(self.dp2(x))

        y = self.conv8(ys2)
        y = self.conv9(y)
        y = self.dp1(y)
        y = self.conv10(y)

        return x, y   #### x is for classification, y is for semantic segmentation
    
if __name__ == '__main__':
    model = DGCNN_cls_semseg()
    xyz = torch.rand(16, 3, 2048)
    model(xyz)

