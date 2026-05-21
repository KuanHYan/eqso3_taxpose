import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- 基础工具：FPS、query ball、索引 --------------------
def farthest_point_sample(xyz, npoint):
    """FPS 采样，xyz: (B, N, 3), 返回采样点索引 (B, npoint)"""
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device).long()
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """按索引取点，points: (B, N, C), idx: (B, S) -> (B, S, C)"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """球查询，xyz: (B,N,3), new_xyz: (B,S,3) -> 分组索引 (B,S,nsample)"""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = torch.sum((new_xyz.view(B, S, 1, 3) - xyz.view(B, 1, N, 3)) ** 2, -1)
    group_idx[sqrdists > radius ** 2] = N   # 超出半径的设为 N（padding）
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # 取前 nsample 个
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]    # 将 padding 点替换为第一个点
    return group_idx


# -------------------- Set Abstraction (编码层) --------------------
class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3  # 坐标作为附加特征
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """xyz: (B,N,3), points: (B,N,C_in) -> new_xyz: (B,S,3), new_points: (B,S,C_out)"""
        B, N, C = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz.view(B, 1, N, 3)   # 所有点为一组
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)

        # 局部坐标归一化（相对于组中心）
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint if not self.group_all else 1, 1, 3)

        if points is not None:
            grouped_points = index_points(points, idx) if not self.group_all else points.view(B, 1, N, -1)
            new_points_fea = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points_fea = grouped_xyz_norm

        # 卷积 + 最大池化
        new_points_fea = new_points_fea.permute(0, 3, 1, 2).contiguous()  # [B, C+3, npoint, nsample]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points_fea = F.relu(bn(conv(new_points_fea)))
        new_points_fea = torch.max(new_points_fea, -1)[0]   # [B, C_out, npoint]
        new_xyz = new_xyz if not self.group_all else xyz[:, :1, :]
        return new_xyz, new_points_fea.permute(0, 2, 1)


# -------------------- Feature Propagation (解码层) --------------------
class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points_fea1, points_fea2):
        """
        将 points2 (来自 xyz2) 传播到 xyz1 并与 points1 (skip connection) 融合
        xyz1: 目标点云 (高分辨率), xyz2: 源点云 (低分辨率)
        points1: 对应 xyz1 的特征 (来自编码器同级，可为 None)
        points2: 对应 xyz2 的特征
        """
        if points_fea2 is not None:
            # 基于距离的插值：k=3, 权重=1/d
            B, N, C = xyz1.shape
            _, S, _ = xyz2.shape
            if S == 1:
                # 直接复制
                interpolated_points = points_fea2.repeat(1, N, 1)
            else:
                dists = torch.sum((xyz1.view(B, N, 1, -1) - xyz2.view(B, 1, S, -1)) ** 2, -1)
                dists, idx = dists.sort(dim=-1)
                dists, idx = dists[:, :, :3], idx[:, :, :3]  # 最近3个邻居
                dist_recip = 1.0 / (dists + 1e-8)
                norm = torch.sum(dist_recip, dim=2, keepdim=True)
                weight = dist_recip / norm
                interpolated_points = torch.sum(index_points(points_fea2, idx) * weight.view(B, N, 3, 1), dim=2)
        else:
            interpolated_points = None

        if points_fea1 is not None:
            # 与跳跃连接拼接
            new_points = torch.cat([points_fea1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # [B, C, N]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)


# -------------------- PointNet2 Autoencoder --------------------
class PointNet2AutoEncoder(nn.Module):
    def __init__(self, feature_dims=[64, 128, 256], D=256):
        super().__init__()
        # 编码器
        self.sa1 = SetAbstraction(512, 0.2, 32, 0, [64, 64, 128])
        self.sa2 = SetAbstraction(128, 0.4, 64, 128, [128, 128, 256])
        self.sa3 = SetAbstraction(None, None, None, 256, [256, 512, D], group_all=True)  # 全局特征 (B,1,D)

        # 解码器 (上采样)
        self.fp3 = FeaturePropagation(256 + D, [256, 256])   # 上采样到 128 点
        self.fp2 = FeaturePropagation(128 + 256, [256, 128]) # 上采样到 512 点
        self.fp1 = FeaturePropagation(0 + 128, [128, 128, 64]) # 上采样到原始点数 (N)

        # 坐标回归头
        self.conv_coord = nn.Conv1d(64, 3, 1)

    def forward(self, xyz):
        B, N, _ = xyz.shape

        # 编码阶段 (保存中间特征和坐标用于 skip connection)
        l1_xyz, l1_points = self.sa1(xyz, None)          # (B, 512, 128)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 128, 256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 1, D)

        # 解码阶段 (逐步上采样回原始坐标)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # 融合 skip: l2
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # 融合 skip: l1
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)          # 恢复到原始 N 点

        # 预测每个点的坐标偏移（或直接预测坐标）
        coords = self.conv_coord(l0_points.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, 3)

        # 返回重建坐标、中间编码特征（可用于下游任务）
        return coords, l3_points.squeeze(1)  # (B, N, 3), (B, D)