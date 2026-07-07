"""上采样 Head: 将稀疏特征映射回稠密点云坐标.

用于 DGCNN_Grouper 降采样场景: backbone 输出 (B,C,M) 稀疏特征,
本模块将其升采样为 (B,3,N) 稠密对应点, 恢复全分辨率 flow 预测.

重构要点 (v2):
  1. 移除内部 attn_proxy — scores 由外部提供
  2. P_coarse 可由外部传入, 传入时跳过 coarse_mlp
  3. P_coarse 初始形状统一为 (B, 3, M) (稀疏分辨率)
  4. 使用 FoldingNet 风格完成 M→N 升采样

设计借鉴 PoinTr/models/PoinTr.py 的 Fold 模块:
  - Fold 将 2D 网格折叠为以粗点为中心的 3D 局部 patch (两步折叠)
  - 本模块: 全局特征 + 可学习 grid → folding1 → cat(fd1, feat) → folding2 → offsets
  - scores (B,N,N) 存在时: 全局池化 → expand N → scores 加权混合
  - scores 不存在时: 纯全局池化 → expand N → 直接折叠
  - 粗点 P_coarse (B,3,M) 通过 1D 插值升到 (B,3,N), 作为偏移的基点
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from taxpose.nets.raw_dgcnn import LayerNorm1d


class LearnedUpsamplingHead(nn.Module):
    """从 (B,C,M) 特征 + (B,3,M) 稀疏坐标, 生成 (B,3,N) 稠密点云.

    流程 (FoldingNet 风格, 两步折叠):
      1. coarse_mlp: feat → P_coarse (B,3,M)    [可选: 外部传入则跳过]
      2. 全局特征: max_pool(feat) → (B,C) → expand → (B,C,N)
         [scores 存在时: 再用 scores(B,N,N) 做特征混合]
      3. 粗点展开: P_coarse (B,3,M) → interpolate → (B,3,N)
      4. 两步折叠: grid+feat → folding1 → cat(fd1,feat) → folding2 → offsets
      5. output = base_pts + offsets * scale
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 128,
        up_ratio: int = 4,
        k: int = 8,                # 保留参数以兼容旧接口, 当前未使用
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.k = k

        # ── 1. 粗坐标生成: feat (B,C,M) → P_coarse (B,3,M) ──
        self.coarse_mlp = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, 3, 1),
        )

        # ── 2. FoldingNet 两步折叠 ──
        #     可学习 2D folding grid: (1, 2, N)
        self.folding_grid = nn.Parameter(
            torch.randn(1, 2, up_ratio) * 0.01)

        # folding1: 第一次折叠  (C + 2) → mid → mid//2 → 3
        # 借鉴 Fold.folding1
        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channels + 2, mid_channels, 1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, mid_channels // 2, 1),
            nn.BatchNorm1d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels // 2, 3, 1),
        )

        # folding2: 第二次折叠  (C + 3) → mid → mid//2 → 3
        # 借鉴 Fold.folding2: 将第一次折叠结果与特征再次拼接
        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channels + 3, mid_channels, 1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, mid_channels // 2, 1),
            nn.BatchNorm1d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels // 2, 3, 1),
        )

        # 可学习缩放因子, 控制位移幅度
        self.scale = nn.Parameter(torch.ones(1, 3, 1) * init_scale)

        self.reduce_map = nn.Conv1d(in_channels*2+3, in_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        feat: torch.Tensor,              # (B, C, M)  稀疏特征
        scores: torch.Tensor = None,     # (B, 1, M)  外部关注度权重
        P_coarse: torch.Tensor = None,   # (B, 3, M)  外部粗坐标 (可选)
        return_coarse: bool = False,
    ):
        """前向传播.

        Args:
            feat:        (B, C, M)  backbone 输出的稀疏特征
            sparse_pts:  (B, 3, M)  稀疏点坐标 (降采样后的坐标)
            scores:      (B, 1, M)  外部M 个稀疏点之间的关注度权重
                         若为 None, 使用纯全局池化.
            P_coarse:    (B, 3, M)  外部提供的粗坐标.
                         若为 None, 由 coarse_mlp 内部生成.
            return_coarse: 是否返回粗坐标.

        Returns:
            output:      (B, 3, N)  升采样后的稠密对应点
            [coarse_pts]: (B, 3, M)  粗坐标 (return_coarse=True 时)
        """
        B, C, M = feat.shape

        # ── 1. 粗坐标生成 (或跳过) → (B, 3, M) ──
        if P_coarse is None:
            P_coarse = self.coarse_mlp(feat)                    # (B, 3, M)
        base_pts = P_coarse.transpose(1, 2).unsqueeze(-1)       # (B, M, 3, 1)

        # ── 2. 全局特征: M 稀疏特征 → 全局池化 → expand 到 N ──
        #     借鉴 PoinTr: global_feature = max_pool(increase_dim(q))

        if scores is not None:
            # scores: (B, 1, M) 各个点代理的关注度权重
            # agg_feat[b, c, n] = ∑_m feat[b, c, m] * scores[b, n, m]
            global_feat = (feat @ scores.transpose(1, 2))                    # (B, C, M)
        else:
            # Fallback: 纯全局池化 → 重复到 N 个点
            global_feat = torch.max(feat, dim=2, keepdim=True)[0]            # (B, C, M)
        agg_feat = torch.cat([feat, global_feat.expand(-1, -1, M), P_coarse], dim=1)   # (B, 2C+3, M)
        agg_feat = self.reduce_map(agg_feat).transpose(1, 2)                 # (B, M, C)
        agg_feat = agg_feat.reshape(-1, C, 1).expand(-1, -1, self.up_ratio)  # (BM, C, S)
        
        # ── 3. 两步折叠 (完全借鉴 Fold.forward) ──
        #     Step 1: grid + feature → folding1 → fd1
        #     Step 2: cat(fd1, feature) → folding2 → fd2 (最终偏移)
        grid = self.folding_grid.expand(B*M, -1, -1)                 

        fold_in = torch.cat([grid, agg_feat], dim=1)                     # (BM, C+2, S)
        fd1 = self.folding1(fold_in)                                     # (BM, 3, S)
        fold_in = torch.cat([fd1, agg_feat], dim=1)                      # (BM, C+3, S)
        offsets = self.folding2(fold_in).reshape(B, M, 3, -1)            # (B, M, 3, S)

        # ── 5. 粗点 + 折叠偏移 → 最终输出 ──
        #     借鉴 PoinTr: rebuild_points = relative_xyz + coarse_point_cloud.unsqueeze(-1)
        output = base_pts + offsets * self.scale                         # (B, M, 3, S)
        output = output.transpose(-1, -2).reshape(B, -1, 3)              # (B, N, 3)

        if return_coarse:
            return output, base_pts.squeeze(-1)
        return output


class Fold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2


# ---------------------------------------------------------------------------
# 快速测试
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, C, M = 2, 256, 128
    up_ratio = 4
    N_expected = M * up_ratio  # 每个 query 点生成 up_ratio 个局部点

    feat = torch.randn(B, C, M, device=device)
    model = LearnedUpsamplingHead(
        in_channels=C, mid_channels=128, up_ratio=up_ratio,
    ).to(device)

    # ── Test 1: 最简调用, 无 scores, 无 P_coarse ──
    out = model(feat)
    print(f"Test 1 (no scores, no P_coarse): out.shape = {out.shape}")
    assert out.shape == (B, N_expected, 3), \
        f"Expected {(B, N_expected, 3)}, got {out.shape}"

    # ── Test 2: 带 scores (B, 1, M) + return_coarse ──
    scores = torch.softmax(torch.randn(B, 1, M, device=device), dim=-1)
    out, coarse = model(feat, scores=scores, return_coarse=True)
    print(f"Test 2 (with scores B×1×M):  out={out.shape}, coarse={coarse.shape}")
    assert out.shape == (B, N_expected, 3)
    assert coarse.shape == (B, M, 3)

    # ── Test 3: 外部提供 P_coarse (跳过 coarse_mlp) ──
    ext_coarse = torch.randn(B, 3, M, device=device)
    out = model(feat, P_coarse=ext_coarse)
    print(f"Test 3 (external P_coarse): out.shape = {out.shape}")
    assert out.shape == (B, N_expected, 3)

    # ── Test 4: scores + 外部 P_coarse, 验证 P_coarse 不被修改 ──
    out, coarse = model(
        feat, scores=scores, P_coarse=ext_coarse, return_coarse=True,
    )
    print(f"Test 4 (scores + ext coarse): out={out.shape}, coarse={coarse.shape}")
    assert out.shape == (B, N_expected, 3)
    # return_coarse 返回 (B,M,3), ext_coarse 是 (B,3,M)
    assert coarse.shape == (B, M, 3)
    print(f"  coarse == ext_coarse^T: {torch.allclose(coarse, ext_coarse.transpose(1, 2))}")

    # ── Test 5: 梯度回传验证 ──
    feat_grad = torch.randn(B, C, M, device=device, requires_grad=True)
    out_grad = model(feat_grad)
    loss = out_grad.sum()
    loss.backward()
    print(f"Test 5 (gradient flow): feat.grad is not None = {feat_grad.grad is not None}")
    assert feat_grad.grad is not None, "Gradient should flow back to input feat"

    # ── Test 6: 不同 M (稀疏点数) 的兼容性 ──
    for test_M in [64, 256, 512]:
        test_feat = torch.randn(B, C, test_M, device=device)
        out = model(test_feat)
        exp_N = test_M * up_ratio
        print(f"Test 6 (M={test_M:>3}): out.shape={out.shape}, expected N={exp_N}")
        assert out.shape == (B, exp_N, 3), \
            f"M={test_M}: expected {(B, exp_N, 3)}, got {out.shape}"

    # ── Test 7: scores 的 softmax 归一化不影响数值稳定性 ──
    scores_raw = torch.randn(B, 1, M, device=device) * 10  # 大值
    scores_soft = torch.softmax(scores_raw, dim=-1)
    out_raw = model(feat, scores=scores_raw)
    out_soft = model(feat, scores=scores_soft)
    print(f"Test 7 (scores stability): "
          f"raw range=[{out_raw.min():.4f}, {out_raw.max():.4f}], "
          f"soft range=[{out_soft.min():.4f}, {out_soft.max():.4f}]")

    # ── Test 8: folding_grid 确实被优化 (require_grad) ──
    assert model.folding_grid.requires_grad, "folding_grid should be learnable"
    print(f"Test 8 (folding_grid learnable): shape={model.folding_grid.shape}, "
          f"requires_grad={model.folding_grid.requires_grad}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ All tests passed!  Total params: {n_params:,}")
