import torch
from torch import nn
from pointnet2_ops import pointnet2_utils
# from knn_cuda import KNN
# knn = KNN(k=16, transpose_mode=False)


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist  


class DGCNN_Grouper(nn.Module):
    def __init__(self, emb_dims=256, output_num=256, knn=16, dropout=0.1, norm='BN'):
        super().__init__()
        '''
        K has to be 16
        '''
        self.k = knn
        self.emb_dims = emb_dims
        self.norm = norm
        if norm == 'BN':
            norm1 = nn.BatchNorm2d(64)
            norm2 = nn.BatchNorm2d(256)
            norm3 = nn.BatchNorm2d(1024)
        elif norm == 'GN':
            norm1 = nn.GroupNorm(4, 64)
            norm2 = nn.GroupNorm(4, 256)
            norm3 = nn.GroupNorm(4, 1024)
        else:
            raise ValueError('Invalid normalization: %s' % norm)

        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=1, bias=False),
                                    norm1,
                                    nn.LeakyReLU(negative_slope=0.2))

        self.layer2 = nn.Sequential(nn.Conv2d(72*2, 256, kernel_size=1, bias=False),
                                    norm2,
                                    nn.LeakyReLU(negative_slope=0.2))

        self.layer3 = nn.Sequential(nn.Conv2d(256*2, 1024, kernel_size=1, bias=False),
                                    norm3,
                                    nn.LeakyReLU(negative_slope=0.2))

        # self.layer4 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
        #                             nn.GroupNorm(4, 256),
        #                             nn.LeakyReLU(negative_slope=0.2))
        fpn_dim = 8 + 64 + 256 + 1024
        self.conv5 = nn.Sequential(
            nn.Conv1d(fpn_dim, emb_dims, kernel_size=1, bias=False),
        )
        # self.output_act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.output_num = output_num

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k, k=16):

        # coor: bs, 3, np, x: bs, c, np

        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        # _, idx = knn(coor_k, coor_q)  # bs k np
        idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
        idx = idx.transpose(-1, -2).contiguous()
        assert idx.shape[1] == k
        idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
        idx = idx + idx_base
        idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x):

        # x: bs, 3, np
        coor = x
        f1 = self.input_trans(x)

        f = self.get_graph_feature(coor, f1, coor, f1, self.k)
        f = self.dropout(self.layer1(f))
        f2 = f.max(dim=-1, keepdim=False)[0]  # bs, 64, np//2, k -> bs, 64, np//2

        f_ = torch.cat([f1, f2], dim=1)

        coor_q, f_q = self.fps_downsample(coor, f_, self.output_num)
        f = self.get_graph_feature(coor_q, f_q, coor, f_, self.k)
        f = self.dropout(self.layer2(f))
        f3 = f.max(dim=-1, keepdim=False)[0]  # bs, 128, onp, k -> bs, 64, onp
        coor = coor_q

        f = self.get_graph_feature(coor, f3, coor, f3, self.k)
        f = self.dropout(self.layer3(f))
        f4 = f.max(dim=-1, keepdim=False)[0]  # bs, 128, onp, k -> bs, 128, onp

        # coor_q, f_q = self.fps_downsample(coor, f, self.output_num)
        # f = self.get_graph_feature(coor_q, f_q, coor, f)
        # f = self.layer4(f)
        # f = self.dropout(f.max(dim=-1, keepdim=False)[0])
        # coor = coor_q

        f = torch.cat([f_q, f3, f4], dim=1)
        f = self.conv5(f)
        # f = self.output_act(f)
        return (f, coor)

if __name__ == '__main__':
    x = torch.rand(2, 3, 1024).cuda()
    grouper = DGCNN_Grouper(emb_dims=512, output_num=512).cuda()
    coor, f = grouper(x)
    print(coor.shape)
    print(f.shape)
