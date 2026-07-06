import torch.nn.functional as F
from torch import nn
import torch
from .gemo_fea import ManualPointWiseGemoFea


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
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm1d, relu_type='relu'):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.norm1 = NormWrapper(out_channels, norm_layer)
        self.norm2 = NormWrapper(out_channels, norm_layer)
        if relu_type == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif relu_type == 'leaky_relu':
            self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # 维度不匹配时需要投影
        if in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, 1, bias=False)
            self.proj_norm = NormWrapper(out_channels, norm_layer)
        else:
            self.proj = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.proj is not None:
            residual = self.proj(residual)
            residual = self.proj_norm(residual)
        out = out + residual          # 残差加在 ReLU 之前
        out = self.relu(out)          # 激活在加法之后（标准 ResNet 做法）
        return out


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
                x = norm(x.permute(1, 2)).permute(1, 2)
            else:
                x = norm(x)
            x = F.relu(x)
        return x


class PointwiseMLP(nn.Module):
    def __init__(self, layer_dims, out_dim=1, norm=None):
        super(PointwiseMLP, self).__init__()

        convs = []
        norms = []

        for j in range(len(layer_dims) - 1):
            convs.append(
                nn.Conv1d(layer_dims[j], layer_dims[j + 1], kernel_size=1, bias=False)
            )
            # norms.append(nn.BatchNorm1d(layer_dims[j + 1]))
            if norm is None:
                norms.append(nn.Identity())
            else:
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

    def __init__(self, layer_dims, norm=nn.BatchNorm1d, relu_type='relu'):
        """
        Args:
            layer_dims: [C_in, C_mid, C_out]
            norm: normalization layer
            relu_type: "relu" or "leaky_relu"
        """
        super().__init__()
        self.relu_type = relu_type
        assert relu_type in ['relu', 'leaky_relu'], "Invalid ReLU type"
        self.first_conv_channel = layer_dims[0]
        layer_dims[0] = self.first_conv_channel
        blocks = []
        for i in range(len(layer_dims) - 1):
            in_c = layer_dims[i]
            out_c = layer_dims[i+1]
            blocks.append(ResidualPointNetBlock(in_c, out_c, norm, relu_type))
        self.blocks = nn.Sequential(*blocks)
        # 最后输出层：线性投影到 3 维（点坐标）
        self.output = nn.Conv1d(layer_dims[-1], 3, kernel_size=1, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 对 ReLU 激活使用 Kaiming 初始化
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity=self.relu_type)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, NormWrapper):
                nn.init.constant_(m.norm_layer.weight, 1)
                nn.init.constant_(m.norm_layer.bias, 0)

    def forward(self, x):
        """
        x: (B, C_in, N)
        return: (B, 3, N)
        """
        assert x.shape[1] == self.first_conv_channel
        feat = self.blocks(x)          # (B, C_mid, N)
        out = self.output(feat)        # (B, 3, N)
        return out
