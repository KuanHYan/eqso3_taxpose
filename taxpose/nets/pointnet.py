import torch.nn.functional as F
from torch import nn
import torch


class NormWrapper(nn.Module):
    def __init__(self, num_feature, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.norm_layer = norm_layer(num_feature)

    def forward(self, x):
        if isinstance(self.norm_layer, nn.LayerNorm):
            # B, C, N -> B, N, C -> B, N, C
            x = self.norm_layer(x.swapaxes(1, 2)).swapaxes(1, 2)
        else:
            x = self.norm_layer(x)
        return x


class ResidualPointNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.norm = NormWrapper(out_channels, norm_layer)
        self.relu = nn.ReLU(inplace=True)
        # 维度不匹配时需要投影
        if in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, 1, bias=False)
            self.proj_norm = NormWrapper(out_channels, norm_layer)
        else:
            self.proj = None

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        if self.proj is not None:
            residual = self.proj(residual)
            residual = self.proj_norm(residual)
        return out + residual   # 残差加在 ReLU 后，更利于梯度流通


class PointNet(nn.Module):
    def __init__(self, layer_dims, norm=nn.BatchNorm1d):
        super(PointNet, self).__init__()

        convs = []
        norms = []

        for j in range(len(layer_dims) - 1):
            convs.append(
                nn.Conv1d(layer_dims[j], layer_dims[j + 1], kernel_size=1, bias=False)
            )
            # norms.append(nn.BatchNorm1d(layer_dims[j + 1]))
            norms.append(norm(layer_dims[j + 1]))  # B, C, H, W 对应 B, L, S
            # norms.append(nn.InstanceNorm1d(layer_dims[j + 1]))

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

    def forward(self, x):
        for norm, conv in zip(self.norms, self.convs):
            x = conv(x)  # B, new_c, N
            if isinstance(norm, nn.LayerNorm):
                # B, new_c, N -> B, N, new_c -> B, new_c, N
                x = norm(x.swapaxes(1, 2)).swapaxes(1, 2)
            else:
                x = norm(x)
            x = F.relu(x)
        return x


class PointwiseMLP(nn.Module):
    def __init__(self, layer_dims, out_dim=1, norm=nn.BatchNorm1d):
        super(PointwiseMLP, self).__init__()

        convs = []
        norms = []

        for j in range(len(layer_dims) - 1):
            convs.append(
                nn.Conv1d(layer_dims[j], layer_dims[j + 1], kernel_size=1, bias=False)
            )
            # norms.append(nn.BatchNorm1d(layer_dims[j + 1]))
            norms.append(norm(layer_dims[j + 1]))  # B, C, H, W 对应 B, L, S
            # norms.append(nn.InstanceNorm1d(layer_dims[j + 1]))

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.output = nn.Conv1d(layer_dims[-1], out_dim, kernel_size=1, bias=False)

    def forward(self, x, coarse_points=None):
        for norm, conv in zip(self.norms, self.convs):
            x = conv(x)  # B, new_c, N
            if isinstance(norm, nn.LayerNorm):
                # B, new_c, N -> B, N, new_c -> B, new_c, N
                x = norm(x.swapaxes(1, 2)).swapaxes(1, 2)
            else:
                x = norm(x)
            x = F.relu(x)
        return self.output(x)


class ResidualPointNet(nn.Module):
    pos_enc_dim = 24

    def __init__(self, layer_dims, norm=nn.BatchNorm1d,
                 init_scale=0.01, use_coarse_ps=False, pos_encoding=False):
        super().__init__()
        self.use_coarse_ps = use_coarse_ps
        self.pos_encoding = pos_encoding
        self.first_conv_channel = layer_dims[0] + \
            3 * use_coarse_ps + self.pos_enc_dim * pos_encoding
        layer_dims[0] = self.first_conv_channel
        blocks = []
        for i in range(len(layer_dims) - 1):
            in_c = layer_dims[i]
            out_c = layer_dims[i+1]
            blocks.append(ResidualPointNetBlock(in_c, out_c, norm))
        self.blocks = nn.Sequential(*blocks)
        # 最后输出层：线性投影到 3 维（点坐标）
        self.output = nn.Conv1d(layer_dims[-1], 3, kernel_size=1, bias=True)
        # 可训练缩放因子
        if init_scale > 0:
            self.scale = nn.Parameter(torch.ones(1) * init_scale)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 对 ReLU 激活使用 Kaiming 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, NormWrapper):
                nn.init.constant_(m.norm_layer.weight, 1)
                nn.init.constant_(m.norm_layer.bias, 0)
        # 输出层若使用 bias，可以初始化为 0，scale 已提提供整体尺度

    def forward(self, x, coarse_points=None):
        # x: (B, C_in, N)
        if coarse_points is not None and self.use_coarse_ps:
            cat_fea = [x, coarse_points]
            if self.pos_encoding:
                pe = self.positional_encoding(coarse_points)  # (B, pos_enc_dim, N)
                cat_fea += [pe]
            x = torch.cat(cat_fea, dim=1)
        assert x.shape[1] == self.first_conv_channel
        feat = self.blocks(x)          # (B, C_mid, N)
        out = self.output(feat)        # (B, 3, N)
        if getattr(self, 'scale', None) is not None:
            out = out * self.scale
        return out

    def positional_encoding(self, coords):
        """对坐标做正余弦位置编码"""
        B, _, M = coords.shape
        # coords: (B,3,M)
        pe = []
        for i in range(4):  # 4个频段
            for fn in [torch.sin, torch.cos]:
                pe.append(fn(coords * (2.0**i)))
        pe = torch.cat(pe, dim=1)  # (B, 3*8, M) 这里实际是24维，可控制
        return pe[:, :self.pos_enc_dim, :]  # 截取设定维度