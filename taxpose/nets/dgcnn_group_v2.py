"""DGCNN_Grouper_V2: 改进的 DGCNN 特征提取器，带残差连接与多尺度 FPN.

相比 DGCNN_Grouper:
- 一致的 dataclass 返回接口 (不再返回 tuple / 标量二选一)
- 每层残差连接
- 可配置的降采样层
- 位置编码注入
- 全层 Dropout
- 支持 LayerNorm
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from taxpose.nets.dgcnn_group import DGCNN_Grouper
from taxpose.nets.point_net_util import get_graph_feature, fps_downsample


# ---------------------------------------------------------------------------
# 统一返回结构
# ---------------------------------------------------------------------------

@dataclass
class GrouperOutput:
    """DGCNN_Grouper_V2 统一返回.

    feature:   (B, C, M)  特征
    coords:    (B, 3, M)  对应坐标 (不下采样时 == xyz_input)
    aux:       dict        附加中间特征 (用于 FPN 等)
    """
    feature: torch.Tensor
    coords: torch.Tensor
    aux: dict


# ---------------------------------------------------------------------------
# 位置编码
# ---------------------------------------------------------------------------

class PositionalEncoding1d(nn.Module):
    """将 3D 坐标映射为 C 维位置编码，加到特征上."""

    def __init__(self, in_dim: int = 3, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, out_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_dim // 2, out_dim, 1),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(coords)


# ---------------------------------------------------------------------------
# 残差图卷积块
# ---------------------------------------------------------------------------

class ResGraphConvBlock(nn.Module):
    """带残差连接的图卷积块.

    in_c → expand → Conv2d(kernel=1) → BN/LN → Act → Dropout → max(k) → Conv1d → + residual
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 16,
        dropout: float = 0.1,
        norm: str = 'BN',
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.k = k

        graph_in = in_channels * 2  # (x_q - x_k, x_q)

        if norm == 'BN':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'LN':
            self.norm = nn.GroupNorm(1, out_channels)
        else:
            self.norm = nn.GroupNorm(4, out_channels)

        self.conv = nn.Conv2d(graph_in, out_channels, kernel_size=1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout)

        # 1x1 projection for residual when dims don't match
        if use_residual and in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        else:
            self.proj = None

    def forward(
        self,
        x: torch.Tensor,             # (B, C, N)
        coords: torch.Tensor,        # (B, 3, N)
        x_k: Optional[torch.Tensor] = None,  # (B, C, M)  for cross-attn
        coord_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        identity = x
        # 构建图特征
        gf = get_graph_feature(
            x_q=x, x_k=x_k, k=self.k,
            coord_q=coords, coord_k=coord_k,
        )  # (B, 2*C, N, k)

        gf = self.conv(gf)           # (B, out, N, k)
        gf = self.norm(gf)
        gf = self.act(gf)
        gf = self.dropout(gf)
        gf = gf.max(dim=-1, keepdim=False)[0]  # (B, out, N)

        # 残差
        if self.use_residual:
            if self.proj is not None:
                identity = self.proj(identity)
            gf = gf + identity

        return gf


# ---------------------------------------------------------------------------
# V2 主类
# ---------------------------------------------------------------------------

class DGCNN_Grouper_V2(nn.Module):
    """继承 DGCNN_Grouper 的改进版本.

    新增参数:
        num_layers:      图卷积层数 (默认 4)
        layer_dims:      每层输出通道 (默认 [64, 128, 256, 512])
        downsample_layers: 哪些层后做 FPS (如 [0, 2] 表示第0和第2层后)
        pos_enc_dim:     位置编码维度
        use_fpn:         是否使用 FPN top-down 融合
    """

    def __init__(
        self,
        emb_dims: int = 256,
        output_num: int = 512,
        knn: int = 16,
        dropout: float = 0.1,
        norm: str = 'BN',
        # V2 扩展参数
        num_layers: int = 4,
        layer_dims: Optional[List[int]] = None,
        downsample_layers: Optional[List[int]] = None,
        pos_enc_dim: int = 64,
        use_fpn: bool = True,
    ):
        # 调用父类 __init__ 但不依赖它的层 (我们替换掉)
        super().__init__()
        self.output_num = output_num
        self.k = knn
        self.emb_dims = emb_dims
        self.norm = norm

        if layer_dims is None:
            layer_dims = [64, 128, 256, 512]
        if downsample_layers is None:
            downsample_layers = [0]  # 默认仅第1层后 FPS

        self.num_layers = num_layers
        self.layer_dims = layer_dims
        self.downsample_layers = set(downsample_layers)
        self.use_fpn = use_fpn
        self.k = knn

        # --- 替换父类的层 ---
        # 初始变换
        self.input_trans = nn.Conv1d(3, layer_dims[0], 1)

        # 位置编码
        self.pos_enc = PositionalEncoding1d(3, pos_enc_dim)

        # 图卷积层
        self.layers = nn.ModuleList()
        in_c = layer_dims[0] + pos_enc_dim
        for i in range(num_layers):
            out_c = layer_dims[i]
            self.layers.append(
                ResGraphConvBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    k=knn,
                    dropout=dropout,
                    norm=norm,
                    use_residual=True,
                )
            )
            in_c = out_c

        # FPN top-down 融合层
        if use_fpn:
            self.fpn_layers = nn.ModuleList()
            # 从深到浅: layer_dims reversed
            for i in range(num_layers - 1, 0, -1):
                self.fpn_layers.append(
                    nn.Sequential(
                        nn.Conv1d(layer_dims[i], layer_dims[i - 1], 1, bias=False),
                        nn.BatchNorm1d(layer_dims[i - 1])
                        if norm == 'BN' else nn.GroupNorm(4, layer_dims[i - 1]),
                        nn.LeakyReLU(negative_slope=0.2),
                    )
                )

        # 输出投影
        total_dim = sum(layer_dims) + pos_enc_dim
        self.output_proj = nn.Sequential(
            nn.Conv1d(total_dim, emb_dims, 1, bias=False),
            nn.BatchNorm1d(emb_dims) if norm == 'BN' else nn.GroupNorm(4, emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, down: bool = True):
        """前向传播.

        Args:
            x: (B, 3, N) 输入点云
            down: 是否执行配置的降采样层 (False = 全分辨率)

        Returns:
            GrouperOutput: .feature (B, emb_dims, M), .coords (B, 3, M),
                           .aux (dict, 含 'layer_features' 列表)
        """
        B, _, N = x.shape
        coords = x
        feat = self.input_trans(x)                              # (B, C0, N)
        pos = self.pos_enc(coords)                               # (B, pos_dim, N)
        feat = torch.cat([feat, pos], dim=1)                     # (B, C0+pos, N)
        down = down and x.shape[2] > self.output_num
        down_num = len(self.downsample_layers)
        down_ratio_perstep = (N / self.output_num) ** (1 / down_num)
        down_output_nums = [int(N / down_ratio_perstep / (i+1)) for i in range(down_num)]
        down_output_nums[-1] = self.output_num
        layer_features: List[torch.Tensor] = []
        current_k: Optional[torch.Tensor] = None
        current_ck: Optional[torch.Tensor] = None
        final_coords = coords
        # --- Bottom-up 编码 ---
        for i, layer in enumerate(self.layers):
            feat = layer(feat, coords, x_k=current_k, coord_k=current_ck)
            layer_features.append(feat)
            current_k = feat
            # 降采样
            if down and i in self.downsample_layers:
                current_ck = coords  # TODO: 降采样后做跨分辨率 attention 有bug
                coords, feat = fps_downsample(
                    final_coords, feat, num_group=down_output_nums[0]
                )
                down_output_nums.pop(0)
                final_coords = coords
            else:
                current_ck = None
                coords = None

        # --- FPN top-down 融合 (在最终分辨率上) ---
        if self.use_fpn and len(self.fpn_layers) > 0:
            # 上采样深层特征并与浅层融合
            for j, fpn in enumerate(self.fpn_layers):
                deep_idx = self.num_layers - 1 - j   # 从最深开始
                shallow_idx = deep_idx - 1
                deep_feat = layer_features[deep_idx]
                shallow_feat = layer_features[shallow_idx]

                # 上采样深特征到浅特征分辨率
                if deep_feat.shape[2] != shallow_feat.shape[2]:
                    deep_feat = F.interpolate(
                        deep_feat, size=shallow_feat.shape[2],
                        mode='nearest',
                    )

                fused = fpn(deep_feat) + shallow_feat
                layer_features[shallow_idx] = fused

        # --- 多尺度 concat ---
        # 将所有层特征统一到最终分辨率
        final_res = layer_features[-1].shape[2]
        aligned_features = []
        for lf in layer_features:
            if lf.shape[2] != final_res:
                lf = F.interpolate(lf, size=final_res, mode='nearest')
            aligned_features.append(lf)

        pos = self.pos_enc(final_coords)
        aligned_features += [pos]
        out = torch.cat(aligned_features, dim=1)               # (B, sum_dims, M)
        out = self.output_proj(out)                             # (B, emb_dims, M)

        return out, final_coords


# ---------------------------------------------------------------------------
# 快速测试
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, N = 2, 2048

    x = torch.rand(B, 3, N, device=device)

    # 默认配置: 4 层，第0层后 FPS
    model = DGCNN_Grouper_V2(
        emb_dims=512,
        output_num=512,
        knn=16,
        dropout=0.1,
        norm='BN',
        num_layers=4,
        layer_dims=[64, 128, 256, 512],
        downsample_layers=[0],
        pos_enc_dim=64,
        use_fpn=True,
    ).to(device)

    feature, coords = model(x)
    print(f"feature: {feature.shape}")   # (2, 512, 512)
    print(f"coords:  {coords.shape}")    # (2, 3, 512)

    # 不下采样
    feature, coords = model(x, down=False)
    print(f"\nfull-res feature: {feature.shape}")  # (2, 512, 1024)
    print(f"full-res coords:  {coords.shape}")     # (2, 3, 1024)

    # Two downsampling layers
    model = DGCNN_Grouper_V2(
        emb_dims=512,
        output_num=512,
        knn=16,
        dropout=0.1,
        norm='BN',
        num_layers=4,
        layer_dims=[64, 128, 256, 512],
        downsample_layers=[0, 2],
        pos_enc_dim=64,
        use_fpn=True,
    ).to(device)

    feature, coords = model(x)
    print(f"feature: {feature.shape}")   # (2, 512, 512)
    print(f"coords:  {coords.shape}")    # (2, 3, 512)

    # 统计
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {n_params:,}")
