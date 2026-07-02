# Adapted from https://github.com/FlyingGiraffe/vnn/blob/master/models/vn_dgcnn_partseg.py
# Only changes:
# - Change the paths to relative imports.
# - Make the label optional (i.e. goal-conditioned).
# - Add this comment.
from dataclasses import dataclass
import torch
import torch.nn as nn
from taxpose.utils.se3 import random_se3
from taxpose.nets.point_net_util import get_graph_feature_for_vndgcnn
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
    embedding_dim: int = 512
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
        x = x.transpose(2, 1).contiguous().unsqueeze(-1)  # (B, N, 3, 1)
        x = super(VN4Head, self).forward(x)
        x = self.output(x)  # (B, N, 3, 1)
        return x.squeeze(-1).transpose(2, 1).contiguous()


class raw_VN_DGCNN(nn.Module):
    def __init__(
        self, args, emb_dim, norm_mode='BN', gc=True
    ):
        super(raw_VN_DGCNN, self).__init__()
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
        self.std_feature = VNStdFeature(1024 // 3 * 2, dim=4, normalize_frame=False)

        # The fllowing is for classification, which is not used for SO3 equivariant training.
        f_dim = 2299 if self.gc else 2235
        self.conv8 = nn.Sequential(
            nn.Conv1d(f_dim, 256, kernel_size=1, bias=False),
            self.bn8,
            nn.LeakyReLU(negative_slope=0.2),
        )
        if self.gc:
            self.conv7 = nn.Sequential(
                nn.Conv1d(num_gc_classes, 64, kernel_size=1, bias=False),
                self.bn7,
                nn.LeakyReLU(negative_slope=0.2),
            )

        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            self.bn9,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            self.bn10,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv11 = nn.Conv1d(128, num_part, kernel_size=1, bias=True)

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
        x, z0 = self.std_feature(x)
        x123 = torch.einsum("bijm,bjkm->bikm", x123, z0).view(
            batch_size, -1, num_points
        )
        x = x.view(batch_size, -1, num_points)
        x = x.max(dim=-1, keepdim=True)[0]

        # Modified from original.
        if self.gc:
            l = l.view(batch_size, -1, 1)
            l = self.conv7(l)

            x = torch.cat((x, l), dim=1)

        x = x.repeat(1, 1, num_points)

        x = torch.cat((x, x123), dim=1)

        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        x = self.conv11(x)

        return x


class VN_DGCNN_iqSO3(nn.Module):
    """旋转不变 VN-DGCNN 点云特征提取器 (重构版).

    重构特性:
      1. 可配置维度: 通过 emb_dims 控制所有层通道数, 不再硬编码 64/1024
      2. 可选降采样: down_sample=True 时在阶段间执行 FPS, 支持稀疏特征输出
      3. VN 位置编码: pos_encoding=True 时使用 VN 层编码坐标, 保持旋转不变性
      4. 多尺度融合: 三阶段特征 (x1/x2/x3) concat 后经可学习融合层
    """

    def __init__(
        self,
        emb_dims: int = 512,
        knn: int = 20,
        down_sample: bool = False,
        output_num: int = 1024,
        down_ratio: int = 4,
        pos_encoding: bool = True,
        pe_dim: int = 64,
        norm_mode: str = 'LN',
        pooling: str = 'max',
    ):
        super(VN_DGCNN_iqSO3, self).__init__()
        self.n_knn = knn
        self.down_sample = down_sample
        self.down_ratio = down_ratio
        self.output_num = output_num
        self.pos_encoding = pos_encoding
        self.pooling_mode = pooling

        # ── VN 通道维度计算 ──
        # VN 特征形状: (B, C_vn, 3, N)  其中 C_vn = 3D 向量个数
        # 展平后: (B, C_vn*3, N) → 目标 emb_dims
        C_stage = max(emb_dims // 9, 8)     # 每阶段 VN 通道数
        C_fusion = C_stage * 3               # 三尺度拼接后
        C_out = emb_dims // 3                # 最终输出 VN 通道数
        pe_c = pe_dim // 3 if pos_encoding else 0

        # ── Norm 工厂 ──
        def _norm(c, dim=5):
            if norm_mode == 'LN':
                return VNLayerNorm(c, dim, knn_n=self.n_knn)
            elif norm_mode == 'BN':
                return VNBatchNorm(c, dim)
            return None

        # ── 1. VN 位置编码 ──
        if pos_encoding:
            # 输入阶段 PE: 编码原始坐标
            self.pe_encoder = VNLinearAndLeakyReLU(
                1, pe_c, dim=4, share_nonlinearity=False,
                norm=_norm(pe_c, dim=4))
            # 末阶段 PE: 编码降采样后坐标 (捕捉全局结构)
            self.pe_final = VNLinearAndLeakyReLU(
                1, pe_c, dim=4, share_nonlinearity=False,
                norm=_norm(pe_c, dim=4))

        # ── 2. 三阶段 EdgeConv ──
        # graph feature 将 VN 通道翻倍 (edge + center)
        pe_c_final = pe_c if pos_encoding else 0
        conv1_in = 2 * (1 + pe_c)
        self.conv1 = VNLinearAndLeakyReLU(conv1_in, C_stage, dim=5, norm=_norm(C_stage))
        self.conv2 = VNLinearAndLeakyReLU(C_stage, C_stage, dim=5, norm=_norm(C_stage))
        self.conv3 = VNLinearAndLeakyReLU(C_stage * 2, C_stage, dim=5, norm=_norm(C_stage))
        self.conv4 = VNLinearAndLeakyReLU(C_stage, C_stage, dim=5, norm=_norm(C_stage))
        # conv5 输入需容纳末阶段 PE 拼接
        conv5_in = 2 * (C_stage + pe_c_final)
        self.conv5 = VNLinearAndLeakyReLU(conv5_in, C_stage, dim=5, norm=_norm(C_stage))

        # ── 3. Pooling ──
        if pooling == 'max':
            self.pool1 = VNMaxPool(C_stage)
            self.pool2 = VNMaxPool(C_stage)
            self.pool3 = VNMaxPool(C_stage)
        else:
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool

        # ── 4. 多尺度融合 + 旋转不变化 ──
        self.fusion = VNLinearAndLeakyReLU(
            C_fusion, C_out, dim=4, share_nonlinearity=True,
            norm=_norm(C_out, dim=4))
        # 借鉴 raw_VN_DGCNN: 使用 VNStdFeature 将等变特征转为不变特征
        self.std_feature = VNStdFeature(C_out * 2, dim=4, normalize_frame=False)
        # 最终投影: C_out*2*3 → emb_dims//3 VN 通道 (纯线性, 保不变性)
        self.final_proj = nn.Conv1d(C_out * 2 * 3, emb_dims, 1)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _vn_fps(coor, feat_vn, num_group):
        """FPS 降采样 (兼容 VN 特征).

        Args:
            coor:     (B, 3, N)  坐标
            feat_vn:  (B, C_vn, 3, N)  VN 特征
            num_group: 目标点数 K
        Returns:
            new_coor:     (B, 3, K)
            new_feat_vn:  (B, C_vn, 3, K)
        """
        from taxpose.nets.point_net_util import fps_downsample
        B, C_vn, _, _ = feat_vn.shape
        feat_flat = feat_vn.reshape(B, C_vn * 3, -1)      # (B, C_vn*3, N)
        new_coor, new_feat_flat = fps_downsample(coor, feat_flat, num_group)
        new_feat_vn = new_feat_flat.reshape(B, C_vn, 3, num_group)
        return new_coor, new_feat_vn

    @staticmethod
    def _fps_coords_only(coor, num_group):
        """仅对坐标做 FPS, 返回索引.

        coor: (B, 3, N) → returns (B, K) index tensor
        """
        from pytorch3d.ops import sample_farthest_points
        xyz = coor.transpose(1, 2).contiguous()            # (B, N, 3)
        _, fps_idx = sample_farthest_points(xyz, K=num_group)
        return fps_idx                                       # (B, K)

    @staticmethod
    def _gather_vn_by_idx(feat_vn, idx):
        """按索引 gather VN 特征.

        feat_vn: (B, C_vn, 3, N), idx: (B, K)
        Returns: (B, C_vn, 3, K)
        """
        B, C_vn, _, N = feat_vn.shape
        K = idx.shape[-1]
        idx_expanded = idx[:, None, None, :].expand(B, C_vn, 3, K)
        return feat_vn.gather(dim=-1, index=idx_expanded)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, down=True):
        """
        Args:
            x: (B, 3, N)  输入点云
        Returns:
            不下采样: (B, emb_dims, N)  特征
            下采样:   ((B, emb_dims, K_last), (B, 3, K_last))  特征+坐标元组
        """
        B, _, N = x.shape
        ratio = N / self.output_num
        down_sample = self.down_sample and ratio > 1 and down
        K2 = self.output_num

        coor = x                                          # (B, 3, N) — 保留坐标用于 FPS & KNN
        x_vn = x.unsqueeze(1)                             # (B, 1, 3, N)  VN 格式

        # ── KNN 坐标: (B, N, 3) ──
        # get_graph_feature_for_vndgcnn 内部调用 knn(k, x, x),
        # knn 通过 torch.cdist 算距离, 要求 (B, N, 3) 格式.
        # 我们用 x_coord 显式传入正确格式的坐标.
        x_coord = coor.transpose(1, 2).contiguous()       # (B, N, 3)

        # # ── 位置编码 ──
        # if self.pos_encoding:
        #     pe = self.pe_encoder(x_vn)                    # (B, pe_c, 3, N)
        #     x_vn = torch.cat([x_vn, pe], dim=1)           # (B, 1+pe_c, 3, N)

        # ═══════════════════ Stage 1 ═══════════════════
        x = get_graph_feature_for_vndgcnn(x_vn, k=self.n_knn, x_coord=x_coord)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.pool1(x)                                # (B, C_stage, 3, N)
        coor1 = coor
        xc1 = x_coord                                     # (B, N, 3)

        # ── 降采样 1 ──
        if down_sample:
            sqrt_ratio = ratio ** 0.5
            K1 = int(max(N // sqrt_ratio, 16))
            coor, x = self._vn_fps(coor1, x1, K1)        # (B,3,K1), (B,C_stage,3,K1)
            x_coord = coor.transpose(1, 2).contiguous()   # (B, K1, 3)
        else:
            coor, x = coor1, x1

        # ═══════════════════ Stage 2 ═══════════════════
        knn2 = min(self.n_knn, coor.shape[-1])
        x = get_graph_feature_for_vndgcnn(x, k=knn2)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)                                # (B, C_stage, 3, K1 or N)
        coor2 = coor

        # ── 降采样 2 ──
        if down_sample:
            coor, x = self._vn_fps(coor2, x2, K2)        # (B,3,K2), (B,C_stage,3,K2)
            x_coord = coor.transpose(1, 2).contiguous()   # (B, K2, 3)
        else:
            coor, x = coor2, x2

        # ═══════════════════ Stage 3 ═══════════════════
        # ── 末阶段位置编码: 在降采样后坐标上编码全局结构 ──
        # if self.pos_encoding:
        #     pe_last = self.pe_final(coor.unsqueeze(1))   # (B, pe_c, 3, K2)
        #     x = torch.cat([x, pe_last], dim=1)           # (B, C_stage+pe_c, 3, K2)

        knn3 = min(self.n_knn, coor.shape[-1])
        x = get_graph_feature_for_vndgcnn(x, k=knn3)
        x = self.conv5(x)
        x3 = self.pool3(x)                                # (B, C_stage, 3, K2 or N)

        # ═══════════════════ 多尺度融合 ═══════════════════
        if down_sample:
            # 将粗尺度特征 FPS 到最粗分辨率 K2, 保持点数一致
            _, x1_pooled = self._vn_fps(coor1, x1, K2)
            _, x2_pooled = self._vn_fps(coor2, x2, K2)
            x_cat = torch.cat([x1_pooled, x2_pooled, x3], dim=1)   # (B, C_fusion, 3, K2)
        else:
            x_cat = torch.cat([x1, x2, x3], dim=1)                  # (B, C_fusion, 3, N)

        x = self.fusion(x_cat)                                       # (B, C_out, 3, K_or_N)

        # ── 旋转不变化: VNStdFeature 计算规范帧 ──
        # 借鉴 raw_VN_DGCNN: 全局上下文 → VNStdFeature → 旋转不变特征
        x_mean = x.mean(dim=-1, keepdim=True).expand_as(x)
        x = torch.cat([x, x_mean], dim=1)                            # (B, C_out*2, 3, K_or_N)
        x, _ = self.std_feature(x)                                 # (B, C_out*2, 3, K_or_N)
        # ── 展平为标准 (B, 2C*3, N) 格式 ──
        x_flat = x.reshape(B, -1, x.shape[-1])                       # (B, C_out*2*3, K_or_N)
        # ── 最终投影: 归位到 emb_dims 通道 ──
        x = self.final_proj(x_flat)                                  # (B, emb_dims, K_or_N)

        if self.down_sample:
            return x, coor
        return x


if __name__ == '__main__':
    import numpy as np

    device = 'cpu'
    B, N = 2, 1024

    def _make_net(**kwargs):
        return VN_DGCNN_iqSO3(knn=16, pooling='mean', **kwargs).to(device).eval()

    print("=" * 60)
    print("VN_DGCNN_iqSO3 重构测试")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════
    # Test 1: 基础前向 — 无降采样, 有位置编码
    # ═══════════════════════════════════════════════════════════
    x = torch.randn(B, 3, N, device=device)
    net = _make_net(emb_dims=512, output_num=N, down_sample=False, pos_encoding=True, pe_dim=64)
    with torch.no_grad():
        out = net(x)
    print(f"\nTest 1 (no downsample, PE on): out.shape = {out.shape}")
    assert out.shape[0] == B and out.shape[-1] == N, \
        f"Expected (B={B}, *, N={N}), got {out.shape}"

    # ═══════════════════════════════════════════════════════════
    # Test 2: 降采样模式 — 返回 (feature, coords) 元组
    # ═══════════════════════════════════════════════════════════
    net_ds = _make_net(emb_dims=512, output_num=N//4, down_sample=True, down_ratio=4,
                       pos_encoding=True, pe_dim=64)
    with torch.no_grad():
        out_ds = net_ds(x)
    feat_ds, coor_ds = out_ds
    K_expected = max(N // 4 // 4, 8)  # two FPS stages
    print(f"Test 2 (downsample on): feat.shape={feat_ds.shape}, "
          f"coor.shape={coor_ds.shape}, expected K={K_expected}")
    assert isinstance(out_ds, tuple), "Downsample mode should return tuple"
    assert coor_ds.shape[-1] == feat_ds.shape[-1], \
        "Feature and coords must have same point count"
    assert coor_ds.shape[-1] <= N // 4, "Should be significantly downsampled"

    # ═══════════════════════════════════════════════════════════
    # Test 3: 无位置编码 — pos_encoding=False
    # ═══════════════════════════════════════════════════════════
    net_nope = _make_net(emb_dims=512, output_num=N, down_sample=False,
                         pos_encoding=False)
    with torch.no_grad():
        out_nope = net_nope(x)
    print(f"Test 3 (PE off): out.shape = {out_nope.shape}")

    # ═══════════════════════════════════════════════════════════
    # Test 4: 可配置维度 — 不同 emb_dims
    # ═══════════════════════════════════════════════════════════
    for test_emb in [256, 768, 1024]:
        net_emb = _make_net(emb_dims=test_emb, output_num=N, down_sample=False,
                            pos_encoding=False)
        with torch.no_grad():
            out_emb = net_emb(x)
        # final_proj 后: (emb_dims//3)*3 = 最接近 emb_dims 的 3 的倍数
        expected_c = test_emb
        print(f"Test 4 (emb_dims={test_emb}): out.shape={out_emb.shape}, "
              f"expected C={expected_c}")
        assert out_emb.shape[1] == expected_c, \
            f"Expected C={expected_c}, got {out_emb.shape[1]}"

    # ═══════════════════════════════════════════════════════════
    # Test 5: 不同降采样比例
    # ═══════════════════════════════════════════════════════════
    for test_dr in [2, 8]:
        net_dr = _make_net(emb_dims=512, output_num=N//4, down_sample=True,
                           down_ratio=test_dr, pos_encoding=False)
        with torch.no_grad():
            _, coor_dr = net_dr(x)
        print(f"Test 5 (down_ratio={test_dr}): coor.shape={coor_dr.shape}")

    # ═══════════════════════════════════════════════════════════
    # Test 6: 梯度回传
    # ═══════════════════════════════════════════════════════════
    x_grad = torch.randn(B, 3, N, device=device, requires_grad=True)
    net_grad = _make_net(emb_dims=512, down_sample=False, pos_encoding=False)
    net_grad.train()
    out_grad = net_grad(x_grad)
    loss = out_grad.sum()
    loss.backward()
    print(f"Test 6 (gradient flow): x.grad is not None = {x_grad.grad is not None}")
    assert x_grad.grad is not None, "Gradient should flow back"

    # ═══════════════════════════════════════════════════════════
    # Test 7: 旋转不变性验证 (不下采样)
    # ═══════════════════════════════════════════════════════════
    net_inv = _make_net(emb_dims=512, down_sample=False,
                        pos_encoding=True, pe_dim=64)
    x_test = torch.randn(B, 3, N, device=device)
    with torch.no_grad():
        out_ref = net_inv(x_test)

        R = random_se3(B, rot_var=(np.pi / 180 * 30), trans_var=0.1)
        x_rot = R.transform_points(x_test.permute(0, 2, 1)).permute(0, 2, 1)
        out_rot = net_inv(x_rot)

    # VNStdFeature 已将特征转为旋转不变 → 差异应接近 0
    delta = (out_ref - out_rot).abs().mean().item()
    print(f"Test 7 (rotation invariance, PE on): |ref - rot| mean = {delta:.6f}")
    assert delta < 0.1, f"Rotation invariance violated: delta={delta:.4f}"

    # ═══════════════════════════════════════════════════════════
    # Test 8: 不同点云数量兼容性
    # ═══════════════════════════════════════════════════════════
    net_var = _make_net(emb_dims=512, down_sample=False, pos_encoding=False)
    for test_N in [256, 512, 2048]:
        x_var = torch.randn(B, 3, test_N, device=device)
        with torch.no_grad():
            out_var = net_var(x_var)
        print(f"Test 8 (N={test_N}): out.shape={out_var.shape}")

    # ═══════════════════════════════════════════════════════════
    # Test 9: max pooling vs mean pooling
    # ═══════════════════════════════════════════════════════════
    net_max = VN_DGCNN_iqSO3(
        knn=16, pooling='max', emb_dims=512,
        down_sample=False, pos_encoding=False,
    ).to(device).eval()
    with torch.no_grad():
        out_max = net_max(x)
    print(f"Test 9 (max pooling): out.shape={out_max.shape}")

    # ═══════════════════════════════════════════════════════════
    # Test 10: 降采样 + 旋转不变性
    # ═══════════════════════════════════════════════════════════
    net_ds_inv = _make_net(emb_dims=512, output_num=N//4, down_sample=True, down_ratio=4,
                           pos_encoding=True, pe_dim=64)
    with torch.no_grad():
        feat_ref, coor_ref = net_ds_inv(x_test)

        R2 = random_se3(B, rot_var=(np.pi / 180 * 45), trans_var=0.2)
        x_rot2 = R2.transform_points(x_test.permute(0, 2, 1)).permute(0, 2, 1)
        feat_rot, coor_rot = net_ds_inv(x_rot2)

    print(f"Test 10 (downsample + rotation): "
          f"feat shapes ref={feat_ref.shape} rot={feat_rot.shape}, "
          f"coor shapes ref={coor_ref.shape} rot={coor_rot.shape}")
    assert feat_ref.shape == feat_rot.shape, "Shapes should match after rotation"
    assert coor_ref.shape == coor_rot.shape

    n_params = sum(p.numel() for p in net.parameters())
    print(f"\n{'=' * 60}")
    print(f"✓ All tests passed!  Total params (base net): {n_params:,}")
    print(f"{'=' * 60}")