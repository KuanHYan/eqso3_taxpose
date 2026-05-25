# Adapted from https://github.com/FlyingGiraffe/vnn/blob/master/models/vn_dgcnn_partseg.py
# Only changes:
# - Change the paths to relative imports.
# - Make the label optional (i.e. goal-conditioned).
# - Add this comment.
from dataclasses import dataclass

import torch.nn as nn
from taxpose.utils.se3 import random_se3
from taxpose.nets.raw_dgcnn import get_graph_feature_for_vndgcnn
from taxpose.nets.vn_layers import (
    VNLinearLeakyReLU,
    VNLinearAndLeakyReLU,
    VNMaxPool,
    mean_pool,
    VNStdFeature,
    VNBatchNorm,
    VNLayerNorm,
    VNLinear
)


@dataclass
class VNArgs:
    n_knn: int = 40
    pooling: str = "mean"


class VN4Head(VNLinearAndLeakyReLU):
    def __init__(self, channel, norm_mode: str = 'none', share_nonlinearity=True):
        norm = VNBatchNorm(channel, 4, -1) if norm_mode == 'BN' \
            else VNLayerNorm(channel, 4, -1)
        if norm_mode == 'BN':
            norm = VNBatchNorm(channel, 4, -1)
        elif norm_mode == 'LN':
            norm = VNLayerNorm(channel, 4, -1)
        else:
            norm = None
        super(VN4Head, self).__init__(
            channel, channel, 4, share_nonlinearity=share_nonlinearity, norm=norm)
        self.output = VNLinear(channel, channel)

    def forward(self, x):
        """
        xyz: B, 3, N
        """
        x = x.transpose(2, 1).unsqueeze(-1)  # (B, N, 3, 1)
        x = super(VN4Head, self).forward(x)
        x = self.output(x)  # (B, N, 3, 1)
        return x.squeeze(-1).transpose(2, 1)


class VN_DGCNN_eqSO3(nn.Module):
    def __init__(
        self, args, emb_dim, norm_mode='BN', gc=True
    ):
        super(VN_DGCNN_eqSO3, self).__init__()
        self.args = args
        self.n_knn = args.n_knn
        self.gc = gc
        if norm_mode == 'BN':
            norm = VNBatchNorm(64 // 3, 5)
        elif norm_mode == 'LN':
            norm = VNLayerNorm(64 // 3, 5, self.n_knn)
        else:
            raise ValueError('Invalid normalization mode')

        self.conv1 = VNLinearAndLeakyReLU(2, 64 // 3, norm=norm)
        self.conv2 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, norm=norm)
        self.conv3 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, norm=norm)
        self.conv4 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, norm=norm)
        self.conv5 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, norm=norm)

        if args.pooling == "max":
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
        elif args.pooling == "mean":
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool

        self.conv6 = VNLinearAndLeakyReLU(
            64 // 3 * 3, 1024 // 3, dim=4, share_nonlinearity=True, norm=VNLayerNorm(1024 // 3, 4, self.n_knn)
        )

    def forward(self, x, l=None):
        """
        x: BxCxN
        return: Bxnum_partxN
        """
        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1)

        x = get_graph_feature_for_vndgcnn(x, k=self.n_knn)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.pool1(x)

        x = get_graph_feature_for_vndgcnn(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature_for_vndgcnn(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)

        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        # x, z0 = self.std_feature(x)
        # x123 = torch.einsum("bijm,bjkm->bikm", x123, z0).view(
        #     batch_size, -1, num_points
        # )
        # x = x.view(batch_size, -1, num_points)
        # x = x.max(dim=-1, keepdim=True)[0]

        # # Modified from original.
        # if self.gc:
        #     l = l.view(batch_size, -1, 1)
        #     l = self.conv7(l)

        #     x = torch.cat((x, l), dim=1)

        # x = x.repeat(1, 1, num_points)

        # x = torch.cat((x, x123), dim=1)

        # x = self.conv8(x)
        # x = self.dp1(x)
        # x = self.conv9(x)
        # x = self.dp2(x)
        # x = self.conv10(x)
        # x = self.conv11(x)

        return x


class VN_DGCNN_iqSO3(nn.Module):
    def __init__(
        self, args, num_part=50, normal_channel=False, gc=True, num_gc_classes=16
    ):
        super(VN_DGCNN_iqSO3, self).__init__()
        self.args = args
        self.n_knn = args.n_knn
        self.gc = gc
        self.num_gc_classes = num_gc_classes

        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        norm = VNLayerNorm(64 // 3, 5, self.n_knn)
        self.conv1 = VNLinearAndLeakyReLU(2, 64 // 3, norm=norm)
        self.conv2 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, norm=norm)
        self.conv3 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, norm=norm)
        self.conv4 = VNLinearAndLeakyReLU(64 // 3, 64 // 3, norm=norm)
        self.conv5 = VNLinearAndLeakyReLU(64 // 3 * 2, 64 // 3, norm=norm)

        if args.pooling == "max":
            self.pool1 = VNMaxPool(64 // 3)
            self.pool2 = VNMaxPool(64 // 3)
            self.pool3 = VNMaxPool(64 // 3)
        elif args.pooling == "mean":
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool

        self.conv6 = VNLinearAndLeakyReLU(
            64 // 3 * 3, 1024 // 3, dim=4, share_nonlinearity=True, norm=VNLayerNorm(1024 // 3, 4, self.n_knn)
        )
        self.std_feature = VNStdFeature(1024 // 3 * 2, dim=4, normalize_frame=False)

        # # The fllowing is for classification, which is not used for SO3 equivariant training.
        # f_dim = 2299 if self.gc else 2235
        # self.conv8 = nn.Sequential(
        #     nn.Conv1d(f_dim, 256, kernel_size=1, bias=False),
        #     self.bn8,
        #     nn.LeakyReLU(negative_slope=0.2),
        # )
        # if self.gc:
        #     self.conv7 = nn.Sequential(
        #         nn.Conv1d(num_gc_classes, 64, kernel_size=1, bias=False),
        #         self.bn7,
        #         nn.LeakyReLU(negative_slope=0.2),
        #     )

        # self.dp1 = nn.Dropout(p=0.5)
        # self.conv9 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=1, bias=False),
        #     self.bn9,
        #     nn.LeakyReLU(negative_slope=0.2),
        # )
        # self.dp2 = nn.Dropout(p=0.5)
        # self.conv10 = nn.Sequential(
        #     nn.Conv1d(256, 128, kernel_size=1, bias=False),
        #     self.bn10,
        #     nn.LeakyReLU(negative_slope=0.2),
        # )
        # self.conv11 = nn.Conv1d(128, num_part, kernel_size=1, bias=True)

    def forward(self, x, l=None):
        """
        x: BxCxN
        return: Bxnum_partxN
        """
        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1)

        x = get_graph_feature_for_vndgcnn(x, k=self.n_knn)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.pool1(x)

        x = get_graph_feature_for_vndgcnn(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)

        x = get_graph_feature_for_vndgcnn(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)

        x123 = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x123)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        # x, z0 = self.std_feature(x)
        # x123 = torch.einsum("bijm,bjkm->bikm", x123, z0).view(
        #     batch_size, -1, num_points
        # )
        # x = x.view(batch_size, -1, num_points)
        # x = x.max(dim=-1, keepdim=True)[0]

        # # Modified from original.
        # if self.gc:
        #     l = l.view(batch_size, -1, 1)
        #     l = self.conv7(l)

        #     x = torch.cat((x, l), dim=1)

        # x = x.repeat(1, 1, num_points)

        # x = torch.cat((x, x123), dim=1)

        # x = self.conv8(x)
        # x = self.dp1(x)
        # x = self.conv9(x)
        # x = self.dp2(x)
        # x = self.conv10(x)
        # x = self.conv11(x)

        return x


if __name__ == '__main__':
    import numpy as np
    # import random
    # random.seed(0)
    # torch.manual_seed(1314)
    x = torch.randn(2, 3, 1024)
    net = VN_DGCNN_eqSO3(VNArgs(n_knn=16), emb_dim=64, norm_mode='LN', gc=False)
    net.eval()
    with torch.no_grad():
        out = net(x)
        print(out.shape)

        R = random_se3(x.shape[0], (np.pi/180*2), 0.0)
        x = R.transform_points(x.permute(0, 2, 1)).permute(0, 2, 1)
        out_so3 = net(x)
        print(out_so3.shape)
        out = R.transform_points(out.transpose(-2, -1).reshape(x.shape[0], -1, 3)).reshape(x.shape[0], -1, x.shape[-1], 3).transpose(-2, -1)
        print(out_so3.shape)
        delta = out - out_so3
        print(delta.mean())