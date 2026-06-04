from dataclasses import dataclass
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


class LayerNorm1d(nn.Module):
    """对 (B, C, L) 输入在 C 维度执行 LayerNorm，保持输出形状不变。"""
    def __init__(self, num_channels, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C) -> LN -> (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).contiguous()
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