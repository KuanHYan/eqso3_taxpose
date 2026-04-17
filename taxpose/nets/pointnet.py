import torch.nn.functional as F
from torch import nn


class PointNet(nn.Module):
    def __init__(self, layer_dims=[3, 64, 64, 64, 128, 512]):
        super(PointNet, self).__init__()

        convs = []
        norms = []

        for j in range(len(layer_dims) - 1):
            convs.append(
                nn.Conv1d(layer_dims[j], layer_dims[j + 1], kernel_size=1, bias=False)
            )
            # norms.append(nn.BatchNorm1d(layer_dims[j + 1]))
            norms.append(nn.LayerNorm(layer_dims[j + 1]))  # B, C, H, W 对应 B, L, S
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
