from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils import weight_norm
from taxpose.nets.pointnet import ResidualPointNet, PointwiseMLP
from taxpose.nets.huggingface_tf import Transformer
from taxpose.nets.transformer_flow_pm import CustomTransformer
from taxpose.nets.vn_dgcnn import VN4Head
from taxpose.nets.moe_wab import MOELayer


def normalize(points, eps=1e-8):
    """
    points: (B, N, 3) 点云batch
    returns:
        normalized: (B, N, 3) 归一化后的点云
        center: (B, 1, 3) 每个样本的重心
        radius: (B, 1, 1) 每个样本的缩放半径（最大距离）
    """
    # 计算重心 (B, 1, 3)
    center = points.mean(dim=1, keepdim=True)  # [B,1,3]
    # 平移至重心
    centered = points - center  # [B,N,3]
    # 计算每个点到原点的距离 (B, N)
    distances = torch.norm(centered, dim=2, keepdim=False)  # [B,N]
    # 最大距离作为半径 (B,1,1)
    radius = distances.max(dim=1, keepdim=True)[0].unsqueeze(-1)  # [B,1,1]
    # 避免除零，若半径为0则置为1
    radius = torch.where(radius < eps, torch.ones_like(radius), radius)
    # 归一化
    normalized = centered / radius  # [B,N,3]
    return normalized, center, radius


def denormalize(normalized_points, center, radius):
    """
    将归一化后的点云恢复到原始坐标空间
    normalized_points: (B, N, 3)
    center: (B, 1, 3)
    radius: (B, 1, 1)
    returns:
        original_points: (B, N, 3)
    """
    return normalized_points * radius + center


class LayerNorm1d(nn.Module):
    """对 (B, C, L) 输入在 C 维度执行 LayerNorm，保持输出形状不变。"""
    def __init__(self, num_channels, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C) -> LN -> (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.norm1 = LayerNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.norm2 = LayerNorm1d(out_channels)
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return F.relu(out)


class LearnedUpsamplingHead(nn.Module):
    def __init__(self, in_channels, mid_channels=128, out_points=1024,
                 k=8, init_scale=0.1, refer_point=False):
        super().__init__()
        self.out_points = out_points
        self.k = k
        self.refer_raw_point = refer_point
        # 注意力打分（用于聚合全局特征）
        self.attn_proxy = nn.Conv1d(in_channels + 3, 1, 1)

        inc = in_channels + int(3*refer_point)
        # 粗坐标生成 MLP（仅依赖特征）
        self.coarse_mlp = nn.Sequential(
            nn.Conv1d(inc, mid_channels, 1, bias=False),
            LayerNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, 3, 1)
        )

        # 特征上采样模块
        self.feat_up_mlp = nn.Sequential(
            nn.Conv1d(in_channels + 3, mid_channels, 1, bias=False),
            LayerNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(mid_channels, mid_channels),
            nn.Conv1d(mid_channels, mid_channels, 1, bias=False),
            LayerNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # 位移解码器
        self.pos_enc_dim = 12
        self.decoder_in = mid_channels + 3 + self.pos_enc_dim
        self.decoder = nn.Sequential(
            nn.Conv1d(self.decoder_in, 128, 1, bias=False),
            LayerNorm1d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128, 128),
            nn.Conv1d(128, 3, 1)
        )

        # 可学习的逐通道缩放因子
        self.scale = nn.Parameter(torch.ones(1, 3, 1) * init_scale)

        # 种子编码，用于生成 M 个不同的粗点（在 __init__ 中注册）
        self.seed_code = nn.Parameter(torch.randn(1, in_channels, out_points) * 0.001)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, LayerNorm1d):
                # LayerNorm 内部有自己默认初始化，也可留空
                pass

    def get_coarse_points(self, feat, points, target_points=None):
        """
        feat: (B, C, N)
        points: (B, 3, N)
        returns: (B, 3, M)
        """
        if self.refer_raw_point:
            assert target_points is not None, "请提供目标点云坐标以生成参考点当refer_raw_point=True"
        B, C, N = feat.shape
        M = self.out_points

        # 拼接特征与坐标，用 softmax 计算各点重要性
        x = torch.cat([feat, points], dim=1)               # (B, C+3, N)
        scores = self.attn_proxy(x)                         # (B, 1, N)
        scores = F.softmax(scores, dim=-1)

        # 加权聚合得到全局特征 (B, C)
        global_feat = torch.sum(feat * scores, dim=2)
        global_feat = global_feat.unsqueeze(2).expand(-1, -1, M)  # (B, C, M)
        # 加种子打破对称性，生成粗坐标
        expanded_feat = global_feat + self.seed_code        # (B, C, M)
        
        if self.refer_raw_point:
            expanded_feat = torch.cat([expanded_feat, target_points], dim=1)
        coarse_coords = self.coarse_mlp(expanded_feat)      # (B, 3, M)
        return coarse_coords

    def forward(self, feat, points, return_coarse=False, target_points=None):
        """
        feat: (B, C, N)  backbone 特征
        points: (B, 3, N) 输入点云坐标
        returns: (B, 3, M) 预测的上采样点云坐标
        """
        B, C, N = feat.shape
        M = self.out_points

        # 1. 生成粗点云坐标
        P_coarse = self.get_coarse_points(feat, points, target_points)     # (B, 3, M)

        # 2. 特征插值（kNN + 反距离加权）
        dists, idx = self.knn(points, P_coarse.detach())    # (B, M, k)

        # 将特征展平：(B, C, N) -> (B*N, C)
        feat_flat = feat.transpose(1, 2).reshape(B * N, C)

        # 构造展平索引
        batch_offset = torch.arange(B, device=feat.device).view(B, 1, 1) * N
        idx = idx + batch_offset                             # (B, M, k)
        idx_flat = idx.reshape(-1, self.k)                   # (B*M, k)

        gathered_feat = feat_flat[idx_flat, :].view(B, M, self.k, C)  # (B, M, k, C)

        # 反距离权重
        weights = 1.0 / (dists + 1e-8)                      # (B, M, k)
        weights = weights / weights.sum(dim=2, keepdim=True) # (B, M, k)
        interpolated_feat = (gathered_feat * weights.unsqueeze(-1)).sum(dim=2)  # (B, M, C)
        interpolated_feat = interpolated_feat.transpose(1, 2)  # (B, C, M)

        # 3. 特征上采样 MLP
        up_input = torch.cat([interpolated_feat, P_coarse], dim=1)  # (B, C+3, M)
        up_feat = self.feat_up_mlp(up_input)                        # (B, mid_channels, M)

        # 4. 位置编码与位移解码
        pe = self.positional_encoding(P_coarse)                     # (B, pos_enc_dim, M)
        dec_in = torch.cat([up_feat, P_coarse, pe], dim=1)
        delta = self.decoder(dec_in)                                # (B, 3, M)

        # 5. 最终坐标
        output = P_coarse + delta * self.scale
        if return_coarse:
            return output, P_coarse
        return output

    def knn(self, xyz1, xyz2):
        """计算 xyz1 (B,3,N) 与 xyz2 (B,3,M) 之间的 k 近邻"""
        B, _, N = xyz1.shape
        _, _, M = xyz2.shape
        dist = torch.cdist(xyz1.transpose(1, 2), xyz2.transpose(1, 2))  # (B, N, M)
        dist, idx = torch.topk(dist, self.k, dim=1, largest=False)      # (B, k, M)
        dist = dist.transpose(1, 2)                                     # (B, M, k)
        idx = idx.transpose(1, 2)                                       # (B, M, k)
        return dist, idx

    def positional_encoding(self, coords):
        """对坐标 (B, 3, M) 生成 sin/cos 位置编码，返回 (B, pos_enc_dim, M)"""
        B, _, M = coords.shape
        pe_list = []
        # 生成 4 个频段，每个频段 sin 与 cos，共 24 维，取前 pos_enc_dim 维
        for i in range(4):
            freq = 2.0 ** i
            pe_list.append(torch.sin(coords * freq))
            pe_list.append(torch.cos(coords * freq))
        pe = torch.cat(pe_list, dim=1)          # (B, 24, M)
        return pe[:, :self.pos_enc_dim, :]      # 截取所需维度


class TransformerHead(nn.Module):
    def __init__(
        self,
        point_encoder_fun=None,
        emb_dims=512,
        output_num=1024,
        pos_enc_dim=0,
        pred_weight=True,
        residual_on=True,
        pos_enc=False,
        norm=nn.BatchNorm1d,
        project_corrs=False,
        project_corrs_mode='mlp'
    ):
        super(TransformerHead, self).__init__()

        self.emb_dims = emb_dims
        self.pred_weight = pred_weight
        self.pos_enc = pos_enc
        if self.pos_enc:
            assert pos_enc_dim > 0
        if self.pred_weight:
            self.proj_flow_weight = PointwiseMLP(
                [emb_dims, emb_dims], 1, norm=None
            )
        self.score = nn.Conv1d(emb_dims + pos_enc_dim, 1, 1)
        self.head_tf = CustomTransformer(
            emb_dims=emb_dims,
            n_blocks=1,
            dropout=0.3,
            ff_dims=4*emb_dims,
            n_heads=emb_dims//64,
            return_attn=True,
            bidirectional=False,
        )
        if point_encoder_fun is None:
            from taxpose.nets.raw_dgcnn import DGCNN4TaxPose
            self.pt_encoder_fun = DGCNN4TaxPose(
                emb_dims, 16, 0.1
            ).forward
        else:
            self.pt_encoder_fun = point_encoder_fun
        
        self.residual_on = residual_on
        self.project_corrs = project_corrs
        if project_corrs and project_corrs_mode == 'mlp':
            self.project_pts = weight_norm(nn.Linear(output_num, output_num, bias=False))
            self.project_bias = weight_norm(nn.Linear(output_num, output_num, bias=False))
        elif project_corrs and project_corrs_mode == 'vn':
            self.project_pts = VN4Head(output_num)
            self.project_bias = VN4Head(output_num)
        elif project_corrs and project_corrs_mode == 'moe':
            self.project_pts = MOELayer(emb_dims, output_num, 16, 1)
            self.project_bias = MOELayer(emb_dims, output_num, 16, 1)

    def forward(self, *input, scores):
        action_embedding = input[0]  # B, C, N
        action_embedding_raw = input[1]
        action_points = input[2]     # B, 3, N
        anchor_points = input[3]
        if action_embedding.shape[2] != action_points.shape[2]:
            # NOTE: 1_dim is for channel dim, 2_dim is for points dim
            action_points = input[4]
            anchor_points = input[5]
        scores = scores.transpose(2, 1).contiguous()
        corr_points = torch.matmul(anchor_points, scores)
        if self.project_corrs:
            corr_points_center = corr_points.mean(dim=2, keepdim=True)
            if not isinstance(self.project_pts, MOELayer):
                inputs = corr_points-corr_points_center
            else:
                inputs = (corr_points-corr_points_center, action_embedding)
            corr_points = self.project_pts(inputs)
            corr_points += corr_points_center

        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points

        # global point is to compute relative vector
        pt_scores = self.score(action_embedding).transpose(2, 1).contiguous()  # B, C, N --> B, N, 1
        pt_scores = F.softmax(pt_scores, dim=1)
        global_pt = corr_points @ pt_scores  # B, 3, 1
        # global_pt = anchor_points.mean(dim=2, keepdim=True)
        # vector from anchor point to global_pt
        global_pt = global_pt.expand_as(anchor_points)

        corr_points_emb = self.pt_encoder_fun(corr_points)
        scores_for_bias, emb_2 = self.head_tf.get_attn_scores(
            action_embedding_raw, corr_points_emb, seq_dim=2)
        scores_for_bias = scores_for_bias.transpose(2, 1).contiguous()  # B, N, M
        residual_flow = torch.einsum("bcn,bnm->bcm", corr_points, scores_for_bias) - global_pt  # B, 3, N
        if self.project_corrs:
            if not isinstance(self.project_pts, MOELayer):
                inputs = residual_flow
            else:
                inputs = (residual_flow, emb_2)
            residual_flow = self.project_bias(inputs)
        
        if self.residual_on:
            flow = residual_flow + corr_flow
        else:
            flow = corr_flow
        
        # flow_mean = flow.mean(dim=1, keepdim=True)
        # flow_centerd = flow - flow_mean
        # flow = flow_centerd * self.scale + flow_mean

        if self.pred_weight:
            weight = self.proj_flow_weight(action_embedding)
            corr_flow_weight = torch.concat([flow, weight], dim=1)
        else:
            corr_flow_weight = flow

        return {
            "full_flow": corr_flow_weight,
            "residual_flow": residual_flow,
            "corr_flow": corr_flow,
            "corr_points": corr_points,
        }


class ResidualMLPHead(nn.Module):
    """
    Base ResidualMLPHead with flow calculated as
    v_i = f(\phi_i) + \tilde{y}_i - x_i
    """

    def __init__(
        self,
        emb_dims=512,
        pred_weight=True,
        residual_on=True,
        norm=nn.BatchNorm1d,
        bias=False,
        use_coarse_ps=False,
        project_corrs=False,
        project_corrs_mode='mlp',
        output_num=1024,
    ):
        super(ResidualMLPHead, self).__init__()

        self.emb_dims = emb_dims
        if use_coarse_ps and residual_on:
            self.proj_flow = ResidualPointNet(
                [emb_dims, emb_dims // 2, emb_dims // 4, emb_dims // 8],
                norm,
                init_scale=0.01,
                use_coarse_ps=True,
                pos_encoding=False,
            )
        elif residual_on:
            self.proj_flow = PointwiseMLP(
                [emb_dims, emb_dims // 2, emb_dims // 4, emb_dims // 8], 3, norm
            )

        self.pred_weight = pred_weight
        if self.pred_weight:
            self.proj_flow_weight = PointwiseMLP(
                [emb_dims, 64, 64, 64, 128, 512], 1, norm
            )
        self.project_corrs = project_corrs
        if project_corrs and project_corrs_mode == 'mlp':
            self.project_pts = weight_norm(nn.Linear(output_num, output_num, bias=False))
        elif project_corrs and project_corrs_mode == 'vn':
            self.project_pts = VN4Head(output_num)
        elif project_corrs and project_corrs_mode == 'moe':
            self.project_pts = MOELayer(emb_dims, output_num, 8, 2)

        self.residual_on = residual_on
        self.use_coarse_ps = use_coarse_ps

    def forward(self, *input, scores, return_embedding=False):
        """
        input:
          action_embedding: B,512,N
          anchor_embedding: B,512,N
          action_points: B,3,N
          anchor_points: B,3,N
          scores: B,N,N, if needed, use this instead of calculating scores
        return:
          dict with keys:
          full_flow: B,3,N
          residual_flow: B,3,N
          corr_flow: B,3,N
          corr_points: B,3,N
          scores: B,N,N
        """
        action_embedding = input[0]
        action_embedding_raw = input[1]  # It's wrong
        action_points = input[2]
        anchor_points = input[3]
        if action_embedding.shape[2] != action_points.shape[2]:
            # NOTE: 1_dim is for channel dim, 2_dim is for points dim
            action_points = input[4]
            anchor_points = input[5]

        assert scores is not None
        # if scores is None:
        #     if len(input) <= 4:
        #         action_query = action_embedding
        #         anchor_key = anchor_embedding
        #     else:
        #         action_query = input[4]
        #         anchor_key = input[5]

        #     d_k = action_query.size(1)
        #     scores = torch.matmul(
        #         action_query.transpose(2, 1).contiguous(), anchor_key
        #     ) / math.sqrt(d_k)
        #     # W_i # B, N, N (N=number of points, 1024 cur)
        #     scores = torch.softmax(scores, dim=2)

        scores = scores.transpose(2, 1).contiguous()
        corr_points = torch.matmul(anchor_points, scores)
        if self.project_corrs:
            corr_points_center = corr_points.mean(dim=2, keepdim=True)
            inputs = corr_points-corr_points_center
            if isinstance(self.project_pts, MOELayer):
                inputs = (inputs, action_embedding)
            corr_points = self.project_pts(inputs)
            corr_points += corr_points_center
        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points

        if self.pred_weight:
            weight = self.proj_flow_weight(action_embedding)

        if self.residual_on:
            if self.use_coarse_ps:
                coarse_pts = corr_points.detach()
                if self.pred_weight:
                    _weight = weight / (
                        weight.sum(dim=1, keepdim=True) + 1e-8
                    )  # 归一化权重和为1
                    center = (coarse_pts * _weight).sum(dim=1, keepdims=True)
                    # distances = torch.norm(centered, dim=2, keepdim=False)  # [B,N]
                    # radius = distances.max(dim=1, keepdim=True)[0].unsqueeze(-1)  # [B,1,1]
                    # radius = torch.where(radius < 1e-8, torch.ones_like(radius), radius)
                    # centered_corr_points = centered / radius
                else:
                    center = coarse_pts.mean(dim=1, keepdim=True)  # [B,1,3]
                # 平移至重心 NOTE: 不再尺度归一化。这可能会带来尺度不匹配问题？
                centered_corr_points = coarse_pts - center  # [B,N,3]
                residual_flow = self.proj_flow(
                    action_embedding, coarse_points=centered_corr_points
                )
                residual_flow = residual_flow + center

            else:
                residual_flow = self.proj_flow(action_embedding)  # B,3,N
                # residual_flow = torch.matmul(residual_flow, scores)
                # anchor_points_i = (
                #     anchor_points.unsqueeze(-1)
                #     .expand(-1, -1, -1, anchor_points.shape[2])
                #     .contiguous()
                # )
                # anchor_points_j = (
                #     anchor_points.unsqueeze(-2)
                #     .expand(-1, -1, anchor_points.shape[2], -1)
                #     .contiguous()
                # )
                # rel_vec = anchor_points_j - anchor_points_i  # B, 3, N, N
                # residual_flow = torch.einsum("bcnm,bmn->bcn", rel_vec, scores)

            flow = residual_flow + corr_flow
        else:
            flow = corr_flow
            residual_flow = torch.zeros_like(flow)

        if self.pred_weight:
            corr_flow_weight = torch.concat([flow, weight], dim=1)
        else:
            corr_flow_weight = flow
        return {
            "full_flow": corr_flow_weight,
            "residual_flow": residual_flow,
            "corr_flow": corr_flow,
            "corr_points": corr_points,
            "scores": scores,
        }


class Emb_dim_256_ResidualMLPHead(nn.Module):
    def __init__(
        self,
        emb_dims=256,
        pred_weight=True,
        residual_on=True,
        norm=nn.BatchNorm1d,
        use_coarse_ps=False,
    ):
        super(Emb_dim_256_ResidualMLPHead, self).__init__()
        self.emb_dims = emb_dims
        if use_coarse_ps:
            self.proj_flow = ResidualPointNet(
                [emb_dims, emb_dims * 4, emb_dims],
                norm,
                init_scale=0.01,
                use_coarse_ps=True,
                pos_encoding=False,
            )
        else:
            self.proj_flow = PointwiseMLP([emb_dims, emb_dims * 4, emb_dims], 3, norm)
        self.pred_weight = pred_weight
        if self.pred_weight:
            self.proj_flow_weight = PointwiseMLP(
                [emb_dims, emb_dims * 4, emb_dims], 1, norm
            )
        self.residual_on = residual_on
        self.use_coarse_ps = use_coarse_ps

    def forward(self, *input, scores=None, return_embedding=False):
        action_embedding = input[0]
        anchor_embedding = input[1]
        action_points = input[2]
        anchor_points = input[3]
        if action_embedding.shape[2] != action_points.shape[2]:
            # NOTE: 1_dim is for channel dim, 2_dim is for points dim
            action_points = input[4]
            anchor_points = input[5]
        if scores is None:
            if len(input) <= 4:
                action_query = action_embedding
                anchor_key = anchor_embedding
            else:
                action_query = input[4]
                anchor_key = input[5]

            d_k = action_query.size(1)
            scores = torch.matmul(
                action_query.transpose(2, 1).contiguous(), anchor_key
            ) / math.sqrt(d_k)
            # W_i # B, N, N (N=number of points, 1024 cur)
            scores = torch.softmax(scores, dim=2)

        scores = scores.transpose(2, 1).contiguous()
        corr_points = torch.matmul(anchor_points, scores)
        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points

        if self.pred_weight:
            weight = self.proj_flow_weight(action_embedding)

        if self.residual_on:
            if self.use_coarse_ps:
                coarse_pts = corr_points.detach()
                if self.pred_weight:
                    _weight = weight / (
                        weight.sum(dim=1, keepdim=True) + 1e-8
                    )  # 归一化权重和为1
                    center = (coarse_pts * _weight).sum(dim=1, keepdims=True)
                else:
                    center = coarse_pts.mean(dim=1, keepdim=True)  # [B,1,3]
                # 平移至重心 NOTE: 不再尺度归一化。这可能会带来尺度不匹配问题？
                centered_corr_points = coarse_pts - center  # [B,N,3]
                residual_flow = self.proj_flow(
                    action_embedding, coarse_points=centered_corr_points
                )
                residual_flow = residual_flow + center

            else:
                residual_flow = self.proj_flow(action_embedding)  # B,3,N
                # residual_flow = torch.matmul(residual_flow, scores)

            flow = residual_flow + corr_flow
        else:
            flow = corr_flow
            residual_flow = torch.zeros_like(flow)

        if self.pred_weight:
            corr_flow_weight = torch.concat([flow, weight], dim=1)
        else:
            corr_flow_weight = flow
        return {
            "full_flow": corr_flow_weight,
            "residual_flow": residual_flow,
            "corr_flow": corr_flow,
            "corr_points": corr_points,
            "scores": scores,
        }


class Coarse_Res_Head(nn.Module):
    def __init__(
        self,
        emb_dims=512,
        output_num=1024,
        pred_weight=True,
        norm=nn.BatchNorm1d,
    ):
        super(Coarse_Res_Head, self).__init__()
        self.emb_dims = emb_dims
        self.proj_flow = LearnedUpsamplingHead(
            emb_dims, out_points=output_num,
            init_scale=0.01, refer_point=True,
            k=16,
        )
        if pred_weight:
            self.proj_flow_weight = PointwiseMLP(
                [emb_dims, 4*emb_dims, emb_dims], 1, norm)
        self.norm = norm
        self.pred_weight = pred_weight

    def forward(self, *input, scores=None, return_embedding=False):
        action_embedding = input[0]
        anchor_embedding = input[1]
        action_points = input[2]
        anchor_points = input[3]
        sample_point = input[4]

        # TODO: set target_points=anchor_points
        corr_points, coarse_pts = self.proj_flow.forward(
            action_embedding,
            sample_point,
            return_coarse=True,
            target_points=anchor_points,
        )
        if self.pred_weight:
            assert action_embedding.size(-1) == corr_points.size(-1)
            weight = self.proj_flow_weight(action_embedding)

        flow = corr_points - action_points
        if self.pred_weight:
            corr_flow_weight = torch.concat([flow, weight], dim=1)
        else:
            corr_flow_weight = flow
        return {
            "full_flow": corr_flow_weight,
            "residual_flow": flow - coarse_pts + action_points,
            "corr_flow": coarse_pts - action_points,
            "corr_points": coarse_pts,
            "scores": scores,
        }


class ResidualMLPHead4RL(ResidualMLPHead):
    def __init__(
        self,
        emb_dims=512,
        pred_weight=True,
        residual_on=True,
        norm=nn.BatchNorm1d,
        bias=False,
        use_coarse_ps=False,
        project_corrs=False,
        project_corrs_mode='mlp',
        output_num=1024,
    ):
        super().__init__(
            emb_dims, pred_weight, 
            residual_on, norm, bias, 
            use_coarse_ps, project_corrs, 
            project_corrs_mode, output_num
        )
        self.corr_pts_std = PointwiseMLP(
            [emb_dims, emb_dims // 2, emb_dims // 4, emb_dims // 8], 3, norm
        )

    def forward(self, *input, scores, return_embedding=False):
        """
        input:
          action_embedding: B,512,N
          anchor_embedding: B,512,N
          action_points: B,3,N
          anchor_points: B,3,N
          scores: B,N,N, if needed, use this instead of calculating scores
        return:
          dict with keys:
          full_flow: B,3,N
          residual_flow: B,3,N
          corr_flow: B,3,N
          corr_points: B,3,N
          scores: B,N,N
        """
        output = super(ResidualMLPHead4RL, self).forward(*input, scores=scores)
        if self.training:
            action_embedding = input[0]
            flow = output["full_flow"][:, :-1, :]
            std = self.corr_pts_std(action_embedding).exp()  # B, 3, N
            pt = Normal(flow, std)
            output["distribution"] = pt

        return output

    def sample(self, *input, scores, return_embedding=False):
        """
        input:
          action_embedding: B,512,N
          anchor_embedding: B,512,N
          action_points: B,3,N
          anchor_points: B,3,N
          scores: B,N,N, if needed, use this instead of calculating scores
        return:
          dict with keys:
          full_flow: B,3,N
          residual_flow: B,3,N
          corr_flow: B,3,N
          corr_points: B,3,N
          scores: B,N,N
        """
        output = super(ResidualMLPHead4RL, self).forward(*input, scores=scores)
        action_embedding = input[0]
        flow = output["full_flow"][:, :-1, :]
        std = self.corr_pts_std(action_embedding).exp()  # B, 3, N
        pt = Normal(flow, std)
        output["distribution"] = pt

        return output


class TransformerHead4RL(TransformerHead):
    def __init__(
        self,
        point_encoder_fun=None,
        emb_dims=512,
        output_num=1024,
        pos_enc_dim=0,
        pred_weight=True,
        residual_on=True,
        pos_enc=False,
        norm=nn.BatchNorm1d,
        project_corrs=False,
        project_corrs_mode='mlp'
    ):
        super().__init__(
            point_encoder_fun, emb_dims, 
            output_num, pos_enc_dim, pred_weight, 
            residual_on, pos_enc, 
            norm, project_corrs,project_corrs_mode 
        )
        self.corr_pts_std = PointwiseMLP(
            [emb_dims, emb_dims // 2, emb_dims // 4, emb_dims // 8], 3, norm
        )

    def forward(self, *input, scores, return_embedding=False):
        """
        input:
          action_embedding: B,512,N
          anchor_embedding: B,512,N
          action_points: B,3,N
          anchor_points: B,3,N
          scores: B,N,N, if needed, use this instead of calculating scores
        return:
          dict with keys:
          full_flow: B,3,N
          residual_flow: B,3,N
          corr_flow: B,3,N
          corr_points: B,3,N
          scores: B,N,N
        """
        output = super(TransformerHead4RL, self).forward(*input, scores=scores)
        if self.training:
            action_embedding = input[0]
            flow = output["full_flow"][:, :-1, :]
            std = self.corr_pts_std(action_embedding).exp()  # B, 3, N
            pt = Normal(flow, std)
            output["distribution"] = pt

        return output

    def sample(self, *input, scores, return_embedding=False):
        """
        input:
          action_embedding: B,512,N
          anchor_embedding: B,512,N
          action_points: B,3,N
          anchor_points: B,3,N
          scores: B,N,N, if needed, use this instead of calculating scores
        return:
          dict with keys:
          full_flow: B,3,N
          residual_flow: B,3,N
          corr_flow: B,3,N
          corr_points: B,3,N
          scores: B,N,N
        """
        output = super(TransformerHead4RL, self).forward(*input, scores=scores)
        action_embedding = input[0]
        flow = output["full_flow"][:, :-1, :]
        std = self.corr_pts_std(action_embedding).exp()  # B, 3, N
        pt = Normal(flow, std)
        output["distribution"] = pt
        return output


@dataclass
class HeadConfig:
    norm: nn.Module = nn.BatchNorm1d
    emb_dims: int = 512
    output_num: int = 1024
    pred_weight: bool = True
    residual_on: bool = True
    head_bias: bool = False
    head_type: str = "residual"
    use_coarse_soft: bool = False
    init_scale: float = 0.01
    up_sample: bool = False
    pos_encoding: bool = False
    project_corrs: bool = False
    project_corrs_mode: str = "mlp"  # "mlp" or "vn"


def create_head(cfg: HeadConfig, embedding_fun=None) -> nn.Module:
    if cfg.up_sample:
        return Coarse_Res_Head(
            cfg.emb_dims,
            cfg.output_num,
            cfg.pred_weight,
            cfg.norm,
        )
    if cfg.head_type == "attention":
        return AttentionHead(
            cfg.emb_dims, cfg.pred_weight, cfg.residual_on, bias=cfg.head_bias
        )
    if cfg.head_type == "transformer":
        return TransformerHead(
            point_encoder_fun=embedding_fun,
            emb_dims=cfg.emb_dims,
            output_num=cfg.output_num,
            pred_weight=cfg.pred_weight,
            residual_on=cfg.residual_on,
            pos_enc=cfg.pos_encoding,
            project_corrs=cfg.project_corrs,
            project_corrs_mode=cfg.project_corrs_mode,
        )
    if cfg.head_type == "mlp_wieght":
        return WeightHead(cfg.emb_dims, cfg.pred_weight, cfg.output_num, cfg.norm)
    # if cfg.head_type == "residual" and cfg.emb_dims == 256:
    #     return Emb_dim_256_ResidualMLPHead(
    #         cfg.emb_dims,
    #         cfg.pred_weight,
    #         cfg.residual_on,
    #         cfg.norm,
    #         cfg.use_coarse_ps,
    #         cfg.project_corrs,
    #         cfg.output_num
    #     )
    if cfg.head_type == "rl_residual":
        return ResidualMLPHead4RL(
            cfg.emb_dims,
            cfg.pred_weight,
            cfg.residual_on,
            cfg.norm,
            cfg.head_bias,
            cfg.use_coarse_soft,
            cfg.project_corrs,
            cfg.project_corrs_mode,
            cfg.output_num,
        )
    if cfg.head_type == "rl_transformer":
        return TransformerHead4RL(
            point_encoder_fun=embedding_fun,
            emb_dims=cfg.emb_dims,
            output_num=cfg.output_num,
            pred_weight=cfg.pred_weight,
            residual_on=cfg.residual_on,
            pos_enc=cfg.pos_encoding,
            project_corrs=cfg.project_corrs,
            project_corrs_mode=cfg.project_corrs_mode,
            norm=cfg.norm
        )
    return ResidualMLPHead(
        cfg.emb_dims,
        cfg.pred_weight,
        cfg.residual_on,
        cfg.norm,
        cfg.head_bias,
        cfg.use_coarse_soft,
        cfg.project_corrs,
        cfg.project_corrs_mode,
        cfg.output_num,
    )


if __name__ == '__main__':
    # model = LearnedUpsamplingHead(256, out_points=1024)
    # model.train()
    # x = torch.randn(3, 3, 512)
    # fea = torch.randn(3, 256, 512)
    # y = model.forward(fea, x)
    # print(y.shape)
    # norm = LayerNorm1d(512)
    # print(isinstance(norm, nn.Module))

    # TFhead
    model = TransformerHead(None, 256, 0, True, True, False)
    query_pt = torch.rand(3, 3, 512)
    query_emb = torch.rand(3, 256, 512)
    tgt_pt = torch.rand(3, 3, 512)
    tgt_emb = torch.rand(3, 256, 512)

    y = model.forward(query_emb, tgt_emb, query_pt, tgt_pt, scores=torch.rand(3, 512, 512))
    print(y.keys())
    print(y['full_flow'].shape)
    print(y['residual_flow'].shape)