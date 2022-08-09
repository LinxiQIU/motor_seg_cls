# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 17:27:37 2022

@author: linux
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_utils import PointNetSetAbstraction, PointNetFeaturePropagation

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
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)
        
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #6*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn7,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn8,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 7, kernel_size=1, bias=False)   #256*7=1792
        
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn9 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn10 = nn.BatchNorm1d(256)
        self.dp3 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, 5)


    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        
        x = get_neighbors(x, k=self.k)         # (batch_size, 6, num_points) -> (batch_size, 6*2, num_points, k)
        x = self.conv1(x)                     # (batch_size, 6*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                     # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_neighbors(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_neighbors(x2, k=self.k)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        
        x = torch.cat((x1, x2, x3), dim=1)     # (batch_size, 64+64+64=192, num_points)
        x4 = self.conv6(x)                    # (batch_size, 192, num_points) -> (batch_size, emb_dims, num_points)
        x = x4.max(dim=-1, keepdim=True)[0]
        x = x.repeat(1, 1, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 64+64+64+emb_dims(1024)=1216, num_points)

        x = self.conv7(x)           # (batch_size, 1216, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)           # (batch_size, 512, num_points) -> (batch_size, 256, num_points)         
        x = self.dp1(x)
        x = self.conv9(x)           # (batch_size, 256, num_points) -> (batch_size, 7, num_points)

        y1 = F.adaptive_max_pool1d(x4, 1).view(batch_size, -1)    # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)    
        y2 = F.adaptive_avg_pool1d(x4, 1).view(batch_size, -1)    # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        y = torch.cat((y1, y2), 1)      # (batch_size, emb_dims*2)

        y = F.leaky_relu(self.bn9(self.linear1(y)), negative_slope=0.2)     # (batch_size, emb_dims*2) -> (batch_size, 512)
        y = self.dp2(y)
        y = F.leaky_relu(self.bn10(self.linear2(y)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)
        y = self.dp3(y)
        y = self.linear3(y)     # (batch_size, 256) -> (batch_size, 5)

        return x, y    # x -> semantic segmentation, y -> classification
