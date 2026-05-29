#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from taxpose.nets.pointnet import PointwiseMLP, ResidualPointNet
from taxpose.nets.vn_layers import (
    VNLinearLeakyReLU,
    VNLinearAndLeakyReLU,
    VNMaxPool,
    mean_pool,
    VNStdFeature,
    VNBatchNorm,
    VNLayerNorm
)


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(-1)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


def get_graph_feature_for_vndgcnn(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(-1)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)
        else:          # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature


class LayerNorm1d(nn.Module):
    """对 (B, C, L) 输入在 C 维度执行 LayerNorm，保持输出形状不变。"""
    def __init__(self, num_channels, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(
            num_channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C) -> LN -> (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


class LayerNorm2d(nn.Module):
    """对 (B, C, L, W) 输入在 C*W 维度执行 LayerNorm，保持输出形状不变。"""
    def __init__(self, num_channels: list, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(
            num_channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        """ 
        x: (B, C, L) -> (B, L, C) -> LN -> (B, L, C) -> (B, C, L)
        """
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class Swapaxes(nn.Module):
    def forward(self, x):
        return x.transpose(-1, -2)


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.knn
        self.norm = args.norm
        if args.norm == 'LN':
            self.bn1 = LayerNorm2d([64, self.k])
            self.bn2 = LayerNorm2d([64, self.k])
            self.bn3 = LayerNorm2d([128, self.k])
            self.bn4 = LayerNorm2d([256, self.k])
            self.bn5 = LayerNorm1d(args.emb_dims)
        elif args.norm == 'BN':
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)
            self.bn5 = nn.BatchNorm1d(args.emb_dims)
        else:
            raise ValueError('Invalid normalization: %s' % args.norm)

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
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


@dataclass
class DGCNNArgs:
    name: str = 'raw_dgcnn'
    knn: int = 20
    emb_dims: int = 512
    dropout: float = 0.3
    norm: str = 'BN'


class DGCNN4TaxPose(DGCNN):
    def __init__(self, emb_dims=512, knn_n=20, dropout=0.1, norm='BN', output_c=1):
        super().__init__(
            DGCNNArgs('raw_dgcnn', knn_n, emb_dims, dropout, norm), output_c)
        self.linear1 = None
        del self.linear1
        self.bn6 = None
        del self.bn6

        self.linear2 = None
        del self.linear2
        self.bn7 = None
        del self.bn7
        self.dp2 = None
        del self.dp2
        self.linear3 = None
        del self.linear3

    def forward(self, x):
        '''
        input: x with shape of [B, 3, N]
        '''
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        return self.dp1(x)


class DGCNN_VAE(nn.Module):
    def __init__(self, args, pos_encoding=False, output_channels=40):
        super(DGCNN_VAE, self).__init__()
        emb_dims = args.emb_dims
        self.encoder = DGCNN4TaxPose(args.emb_dims, args.knn, args.dropout, args.norm)
        if not pos_encoding:
            self.decoder = PointwiseMLP(
                [emb_dims, emb_dims // 2, emb_dims // 4, emb_dims // 8], 3, nn.BatchNorm1d)
        else:
            self.decoder = ResidualPointNet(
                [emb_dims, emb_dims // 2, emb_dims // 4, emb_dims // 8], nn.BatchNorm1d, pos_encoding=True)

    def forward(self, x):
        """
        x with shape of [B, 3, N]
        return: 
            fea with shape of [B, C, N];
            pts with shape of [B, N, 3]
        """
        if self.eval():
            return self.inference(x)
        fea = self.encoder(x)
        pts = self.decoder(fea, x)
        return fea, pts.transpose(-1, -2)

    def process(self, x):
        fea = self.encoder(x)
        pts = self.decoder(fea, x)
        return fea, pts.transpose(-1, -2)

    @torch.no_grad()
    def inference(self, x):
        """
        x with shape of [B, 3, N]
        return: fea with shape of [B, C, N]
        """
        fea = self.encoder(x)
        return fea


@dataclass
class VNArgs:
    knn: int = 16
    emb_dims: int = 512
    dropout: float = 0.3
    norm: str = 'BN'
    pooling: str = "mean"
    channel: int = 3

class VN_DGCNN(nn.Module):
    def __init__(self, args: VNArgs, gc=False):
        super(VN_DGCNN, self).__init__()
        self.args = args
        self.n_knn = args.knn
        self.channel = channel = args.channel
        norm = VNBatchNorm(channel, 4, -1) if args.norm == 'BN' \
            else VNLayerNorm(channel, 4, -1)
        self.conv1 = VNLinearAndLeakyReLU(
            channel, channel, dim=4, norm=norm)
        self.conv2 = VNLinearAndLeakyReLU(
            channel, channel, dim=4, share_nonlinearity=True, norm=norm)
        
    def forward(self, xyz):
        """
        xyz: B, 3, N
        """
        xyz = xyz.transpose(2, 1).unsqueeze(-1)  # (B, N, 3, 1)
        xyz = self.conv2(self.conv1(xyz), )


if __name__ == '__main__':
    torch.cuda.manual_seed(0)
    x = torch.rand(4, 3, 512).cuda()
    score = torch.rand(4, 512, 512).cuda().requires_grad_(True)
    score = F.softmax(score, dim=1)
    score.retain_grad()
    # model = DGCNN_VAE(DGCNNArgs(norm='LN'))
    # y = model(x)
    # print(y[0].shape)
    # print(y[1].shape)
    model = DGCNN4TaxPose().cuda()
    params = torch.load('./taxpose/logs/pretrain_embedding/best_cpkg/new_dgcnn_BN_509.ckpt')['state_dict']
    model.load_state_dict({k.replace('model.emb_nn.', ''): v for k, v in params.items()})
    model.train()

    # wrong case: no_grad
    # sc = x @ score
    # with torch.no_grad():
    #     y = model.forward(sc)
    # loss = (y**2).mean()
    # loss.backward()
    # print(score.grad)

    # case2: requires_grad_(False)
    # model.requires_grad_(False)
    # sc = x @ score
    # y = model.forward(sc)
    # loss = (y**2).mean()
    # loss.backward()

    # case3: detach
    sc = x @ score
    y = model.forward(sc)
    print(y.shape)
    loss = (y**2).mean()
    loss.backward()
    
    
    print(score.grad.shape)

