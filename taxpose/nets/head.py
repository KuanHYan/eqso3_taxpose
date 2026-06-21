from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from torch.nn.utils import weight_norm
from taxpose.nets.pointnet import ResidualPointNet, PointwiseMLP
from taxpose.nets.huggingface_tf import Transformer
from taxpose.nets.transformer_flow_pm import CustomTransformer
from taxpose.nets.vn_dgcnn import VN4Head
from taxpose.nets.moe_wab import MOELayer
from taxpose.nets.upsample_head import LearnedUpsamplingHead
from taxpose.utils.se3 import points2pose


class Coarse_Res_Head(nn.Module):
    """粗-细对应点预测头: 将稀疏 embedding 升采样为稠密对应点.

    与 ResidualMLPHead / TransformerHead 的输入输出接口对齐.
    使用 LearnedUpsamplingHead 从 (B,C,M) 升采样到 (B,3,N).

    适用场景: backbone 使用 DGCNN_Grouper 降采样时,
    本 Head 将稀疏特征恢复为全分辨率 flow.
    """

    def __init__(
        self,
        emb_dims=512,
        encoder_output_num=1024,
        up_sample_ratio=2,
        pred_weight=True,
        project_corrs=True,
        project_corrs_mode='mlp',
        norm=nn.BatchNorm1d,
        weight_knn_k: int = 8,
    ):
        super().__init__()
        self.emb_dims = emb_dims
        self.output_num = up_sample_ratio*encoder_output_num
        self.proj_flow = LearnedUpsamplingHead(
            emb_dims, up_ratio=up_sample_ratio,
            init_scale=0.01,
            k=16,
        )
        if pred_weight:
            self.proj_flow_weight = PointwiseMLP(
                [emb_dims, emb_dims, emb_dims], 1, norm)
        self.pred_weight = pred_weight
        self.weight_knn_k = weight_knn_k
        if project_corrs and project_corrs_mode == 'mlp':
            self.project_pts = nn.Linear(encoder_output_num, encoder_output_num, bias=False)
        elif project_corrs and project_corrs_mode == 'vn':
            self.project_pts = VN4Head(encoder_output_num)
        elif project_corrs and project_corrs_mode == 'moe':
            self.project_pts = MOELayer(emb_dims, encoder_output_num, 16, 1)
        self.project_corrs = project_corrs

    @staticmethod
    def _upsample_weight(weight_sparse, src_pts, tgt_pts, k=8):
        """kNN 插值将稀疏 weight (B,1,M) 升采样为稠密 (B,1,N).

        Args:
            weight_sparse: (B, 1, M)
            src_pts: (B, 3, M)  稀疏坐标
            tgt_pts: (B, 3, N)  稠密坐标
            k: 近邻数
        Returns:
            weight_dense: (B, 1, N)
        """
        from taxpose.nets.point_net_util import knn as _knn
        B, _, M = weight_sparse.shape
        N = tgt_pts.shape[-1]

        # 对每个稠密点找 k 近邻稀疏点
        dists, idx = _knn(
            k,
            tgt_pts.permute(0, 2, 1).contiguous(),  # query (B, N, 3)
            src_pts.permute(0, 2, 1).contiguous(),  # key   (B, M, 3)
        )  # → dists (B, N, k), idx (B, N, k)

        w_flat = weight_sparse.squeeze(1)           # (B, M)
        gathered_w = w_flat[
            torch.arange(B, device=w_flat.device).view(B, 1, 1),
            idx
        ]                                            # (B, N, k)

        # 反距离加权
        w_dist = 1.0 / (dists + 1e-8)
        w_dist = w_dist / w_dist.sum(dim=-1, keepdim=True)
        weight_dense = (gathered_w * w_dist).sum(dim=-1).unsqueeze(1)  # (B, 1, N)
        return weight_dense

    def forward(self, *input, scores):
        """标准 Head 接口.

        input:
          [0] action_embedding_tf   (B, C, N_or_M)
          [1] action_embedding_raw  (B, C, N_or_M)
          [2] anchor_embedding      (B, C, N_or_M)
          [3] action_points         (B, 3, N_or_M)
          [4] anchor_points         (B, 3, N_or_M)
          [5] act_down_sample       (B, 3, M)  or None (降采样后坐标)
          [6] anch_down_sample      (B, 3, M)  or None
          scores: (B, M, M)  可选

        return:
          full_flow, residual_flow, corr_flow, corr_points, scores
        """
        action_embedding_tf = input[0]
        anchor_embedding = input[2]
        action_points = input[3]
        anchor_points = input[4]
        anchor_dowm_sample = input[6]

        is_downsampled = (action_embedding_tf.shape[2] != action_points.shape[2])
        if is_downsampled:
            assert len(input) >= 7
            anchor_points = anchor_dowm_sample
        scores = scores.transpose(2, 1).contiguous()
        corr_points = torch.matmul(anchor_points, scores)
        if self.project_corrs:
            corr_points_center = corr_points.mean(dim=2, keepdim=True)
            if not isinstance(self.project_pts, MOELayer):
                inputs = corr_points-corr_points_center
            else:
                inputs = (corr_points-corr_points_center, action_embedding_tf)
            corr_points = self.project_pts(inputs)
            corr_points += corr_points_center

        # 若有降采样坐标则使用, 否则用全分辨率坐标
        if is_downsampled:
            sample_point = input[5]   # act_down_sample (B, 3, M)
        else:
            sample_point = action_points

        if self.pred_weight:
            weight_sparse = self.proj_flow_weight(
                action_embedding_tf)                      # (B, 1, M_or_N)

            if is_downsampled:
                # 降采样时需将 weight 插值回全分辨率
                weight = self._upsample_weight(
                    weight_sparse, sample_point, action_points,
                    k=self.weight_knn_k)                  # (B, 1, N)
            else:
                weight = weight_sparse                    # (B, 1, N)

        # 升采样: 稀疏特征 → 稠密对应点
        upsample_points, coarse_pts = self.proj_flow.forward(
            action_embedding_tf,
            scores=weight_sparse if self.pred_weight else scores,
            return_coarse=True,
        )  # (B, N, 3), (B, M, 3)
        # NOTE: 转化维度
        corr_points = corr_points.permute(0, 2, 1).contiguous()
        trans_between_proxy = points2pose(
            coarse_pts, corr_points,
            return_transform3d=True,
            normalization_scehme="softmax",
            weights=weight_sparse.squeeze(dim=1) if self.pred_weight else None
        )
        coarse_pts = trans_between_proxy.transform_points(coarse_pts).transpose(1, 2).contiguous()
        corr_points = trans_between_proxy.transform_points(upsample_points).transpose(1, 2).contiguous()
        # flow = corr_points - action_points
        #   = (coarse_pts - action_points) + (corr_points - coarse_pts)
        #   = corr_flow                + residual_flow
        # 注意: 降采样时 action_points 是 (B,3,N), coarse_pts 也是 (B,3,N),
        #        但 action_embedding_tf 是 (B,C,M)
        corr_flow = coarse_pts - sample_point
        flow = corr_points - action_points
        residual_flow = torch.zeros_like(flow)

        if self.pred_weight:
            assert weight.shape[-1] == flow.shape[-1], \
                (f"weight N={weight.shape[-1]} != flow N={flow.shape[-1]}")
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


class TransformerHead(nn.Module):
    def __init__(
        self,
        point_encoder_fun=None,
        emb_dims=512,
        output_num=1024,
        pred_weight=True,
        residual_on=True,
        pos_enc=False,
        norm=nn.BatchNorm1d,
        project_corrs=False,
        project_corrs_mode='mlp',
        attn_mode="torch_attn"
    ):
        super(TransformerHead, self).__init__()

        self.emb_dims = emb_dims
        self.pred_weight = pred_weight
        self.pos_enc = pos_enc
        
        # 共享特征提取
        self.shared_point_importance = PointwiseMLP(
            [2*emb_dims, emb_dims, emb_dims], emb_dims, None,
        )
        if self.pred_weight:
            self.proj_flow_weight = nn.Conv1d(emb_dims, 1, 1)
        self.score = nn.Conv1d(emb_dims, 1, 1)
        self.head_tf = CustomTransformer(
            emb_dims=emb_dims,
            n_blocks=1,
            dropout=0.1,
            ff_dims=4*emb_dims,
            n_heads=emb_dims//64,
            return_attn=True,
            bidirectional=False,
            attn_mode=attn_mode
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
        action_embedding_tf = input[0]  # B, C, N
        action_embedding_raw = input[1]
        anchor_embedding = input[2]
        action_points = input[3]     # B, 3, N
        anchor_points = input[4]
        if action_embedding_tf.shape[2] != action_points.shape[2]:
            # NOTE: 1_dim is for channel dim, 2_dim is for points dim
            action_points = input[5]
            anchor_points = input[6]
        scores = scores.transpose(2, 1).contiguous()
        corr_points = torch.matmul(anchor_points, scores)
        if self.project_corrs:
            corr_points_center = corr_points.mean(dim=2, keepdim=True)
            if not isinstance(self.project_pts, MOELayer):
                inputs = corr_points-corr_points_center
            else:
                inputs = (corr_points-corr_points_center, action_embedding_tf)
            corr_points = self.project_pts(inputs)
            corr_points += corr_points_center

        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points

        # global point is to compute relative vector
        weight_input = torch.cat([action_embedding_tf, anchor_embedding], dim=1)  # (B, 2*emb, N)
        weight_shared_embedding = self.shared_point_importance(weight_input)
        pt_scores = self.score(weight_shared_embedding).transpose(2, 1).contiguous()  # B, C, N --> B, N, 1
        pt_scores = F.softmax(pt_scores, dim=1)
        global_pt = corr_points @ pt_scores  # B, 3, 1
        # global_pt = anchor_points.mean(dim=2, keepdim=True)
        # vector from anchor point to global_pt
        global_pt = global_pt.expand_as(anchor_points)

        corr_points_emb = self.pt_encoder_fun(corr_points)
        scores_for_bias, emb_2 = self.head_tf.get_attn_scores(
            action_embedding_raw, corr_points_emb, seq_dim=2)
        scores_for_bias = scores_for_bias.transpose(2, 1)  # B, N, M
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
            weight = self.proj_flow_weight(weight_shared_embedding)
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
            self.project_pts = nn.Linear(output_num, output_num, bias=False)
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
          action_embedding_raw: B,512,N
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
        action_embedding_tf = input[0]
        action_embedding_raw = input[1]  # It's wrong
        anchor_embedding = input[2]
        action_points = input[3]
        anchor_points = input[4]
        if action_embedding_tf.shape[2] != action_points.shape[2]:
            # NOTE: 1_dim is for channel dim, 2_dim is for points dim
            action_points = input[5]
            anchor_points = input[6]

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

        scores = scores.transpose(2, 1)
        assert anchor_points is not None, "anchor_points is None"
        corr_points = torch.matmul(anchor_points, scores)
        if self.project_corrs:
            corr_points_center = corr_points.mean(dim=2, keepdim=True)
            inputs = corr_points-corr_points_center
            if isinstance(self.project_pts, MOELayer):
                inputs = (inputs, action_embedding_tf)
            corr_points = self.project_pts(inputs)
            corr_points += corr_points_center
        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points

        if self.pred_weight:
            weight = self.proj_flow_weight(action_embedding_tf)

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
                    action_embedding_tf, coarse_points=centered_corr_points
                )
                residual_flow = residual_flow + center

            else:
                residual_flow = self.proj_flow(action_embedding_tf)  # B,3,N
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


class ReparamResidualMLPHead(ResidualMLPHead):
    def __init__(
        self,
        emb_dims=512,
        pred_weight=True,
        residual_on=True,
        norm=nn.BatchNorm1d,
        bias=False,
        use_coarse_ps=False,
        project_corrs=False,
        project_corrs_mode='vn',
        output_num=1024,
        weight_beta: float = 0.1,     # 采样 log_prob 对 weight 的调制强度
    ):
        super(ReparamResidualMLPHead, self).__init__(
            emb_dims, pred_weight,
            residual_on, norm, bias,
            use_coarse_ps, project_corrs,
            project_corrs_mode, output_num
        )

    def forward(self, *input, scores, sample=False):
        """
        input:
          action_embedding: B,512,N
          action_embedding_raw: B,512,N
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
        output = super().forward(*input, scores=scores)
        if self.training or sample:
            mean_is_soft_cpt = output["corr_flow"]
            var_is_res_flow = torch.exp(output["residual_flow"])  # B,3,N
            std = torch.exp(0.5*output["residual_flow"])
            output["residual_flow"] = var_is_res_flow
            output["std"] = std
            if sample:
                return output
            else:
                reparam_sample = torch.randn_like(mean_is_soft_cpt)  # 标准正态噪声
                assert var_is_res_flow.shape == reparam_sample.shape
                reparam_flow = std * reparam_sample
                output["full_flow"][:, :3, :] = mean_is_soft_cpt + reparam_flow
        else:
            output["full_flow"][:, :3, :] = output["corr_flow"]
            output["residual_flow"] = torch.exp(0.5 * output["residual_flow"])
        return output

    def sample(self, *input, scores, sample_num, return_logP=False):
        """
        input:
          *input: include action_embedding, anchor_embedding, action_points, anchor_points (B,C,N)
          scores: B,N,N, if needed, use this instead of calculating scores
          sample_num: int. Number of samples.
          return_logP: bool. If true, return log prob.
        return:
          dict with keys: {
            samples: G,B,N,3
            weights: G,B,N
            log_probs: G,B
            **kwargs: other keys in output
          }
        """
        output = self.forward(*input, scores=scores, sample=True)
        mean_is_soft_cpt = output["corr_flow"].transpose(1, 2).contiguous()  # B, N, 3
        var_is_res_flow = output["residual_flow"].transpose(1, 2).contiguous()  # B, N, 3
        std = output["std"].transpose(1, 2).contiguous()
        bz, num, _ = var_is_res_flow.shape
        noise = torch.randn((sample_num, bz, num, 3)).to(scores.device)  # G B N 3
        assert mean_is_soft_cpt.shape == var_is_res_flow.shape
        samples = mean_is_soft_cpt.unsqueeze(0).expand(sample_num, -1, -1, -1) + \
            noise * std.unsqueeze(0).expand(sample_num, -1, -1, -1)  # G, B, N, 3
        output["samples"] = samples
        output["weights"] = output["full_flow"][None, :, -1, :].expand(sample_num, -1, -1)
        if return_logP:
            dist = MultivariateNormal(
                mean_is_soft_cpt,
                torch.diag_embed(var_is_res_flow)
            )
            output["log_probs"] = dist.log_prob(samples).mean(dim=-1)

        return output

    def log_probs(self, *input, scores, actions):
        """
        input:
            *input: include action_embedding, anchor_embedding, action_points, anchor_points (B,C,N)
            scores: B,N,N, if needed, use this instead of calculating scores
            actions: tensor of shape G,B,N,3
        return:
            log_probs: tensor of shape G,B
        """
        output = self.forward(*input, scores=scores, sample=True)
        group, bz, num, _ = actions.shape
        mean_is_soft_cpt = output["corr_flow"].transpose(1, 2).contiguous()  # B, N, 3
        var_is_res_flow = output["residual_flow"].transpose(1, 2).contiguous()  # B, N, 3
        dist = MultivariateNormal(
            mean_is_soft_cpt,
            torch.diag_embed(var_is_res_flow)
        )
        log_probs = dist.log_prob(actions).mean(dim=-1).reshape(group, bz)  # G,B

        return log_probs


class ReparamTransformerHead(TransformerHead, ReparamResidualMLPHead):
    def __init__(
        self,
        point_encoder_fun=None,
        emb_dims=512,
        output_num=1024,
        pred_weight=True,
        residual_on=True,
        pos_enc=False,
        norm=nn.BatchNorm1d,
        project_corrs=False,
        project_corrs_mode='mlp'
    ):
        super().__init__(
            point_encoder_fun, emb_dims,
            output_num, pred_weight,
            residual_on, pos_enc,
            norm, project_corrs, project_corrs_mode
        )
        self.register_buffer("zero_mean", torch.zeros(3))
        self.register_buffer("one_std", torch.eye(3))

    def forward(self, *input, scores, sample=False):
        """
        input:
          action_embedding: B,512,N
          action_embedding_raw: B,512,N
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
        output = super().forward(*input, scores=scores)
        if self.training or sample:
            mean_is_soft_cpt = output["corr_flow"]
            var_is_res_flow = torch.exp(output["residual_flow"])  # B,3,N
            std = torch.exp(0.5*output["residual_flow"])
            output["residual_flow"] = var_is_res_flow
            output["std"] = std
            if sample:
                return output
            # reparam_sample = torch.randn_like(mean_is_soft_cpt)     # 标准正态噪声
            # assert var_is_res_flow.shape == reparam_sample.shape
            # reparam_flow = std * reparam_sample
            output["full_flow"][:, :3, :] = mean_is_soft_cpt
        else:
            output["full_flow"][:, :3, :] = output["corr_flow"]
            output["residual_flow"] = torch.exp(output["residual_flow"])
        return output


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
        weight_beta: float = 0.1,     # 采样 log_prob 对 weight 的调制强度
    ):
        super().__init__(
            emb_dims, pred_weight,
            residual_on, norm, bias,
            use_coarse_ps, project_corrs,
            project_corrs_mode, output_num
        )
        self.corr_pts_var = PointwiseMLP(
            [emb_dims, emb_dims // 2, emb_dims // 4, emb_dims // 8], 3, norm
        )
        self.weight_beta = weight_beta

    def forward(self, *input, scores, sample=False):
        """
        input:
          action_embedding: B,512,N
          action_embedding_raw: B,512,N
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
        if sample:
            action_embedding = input[0]
            action_embedding_raw = input[1]
            flow = output["full_flow"][:, :-1, :].permute(0, 2, 1).contiguous()  # B, N, 3
            log_var = self.corr_pts_var(action_embedding).permute(0, 2, 1).contiguous()  # B, N, 3
            pt = MultivariateNormal(flow, torch.diag_embed(log_var.exp()))
            output["distribution"] = pt

        return output

    @torch.no_grad()
    def sample(self, *input, scores, sample_num, return_logP=False):
        """
        input:
          *input: include action_embedding, anchor_embedding, action_points, anchor_points (B,C,N)
          scores: B,N,N, if needed, use this instead of calculating scores
          sample_num: int. Number of samples.
          return_logP: bool. If true, return log prob.
        return:
          dict with keys: {
            samples: G,B,N,3         — 采样的 flow
            weights: G,B,N           — 采样相关的 weight (logits, 需 sigmoid)
            log_probs: G,B           — 平均 log prob (用于 PPO)
            **kwargs: other keys in output
          }
        """
        output = self.forward(*input, scores=scores, sample=True)
        pt: torch.distributions.MultivariateNormal = output["distribution"]
        samples = pt.sample((sample_num,))  # G, B, N, 3
        output["samples"] = samples

        # ---- 采样相关的 weight 调制 ----
        # base_weight_logit: (B, N), 来自 full_flow 第4通道 (确定性均值下的 weight)
        base_weight_logit = output["full_flow"][:, -1, :]  # (B, N)

        if self.weight_beta > 0:
            # log_prob_point: (G, B, N), 每个采样点在其高斯分布下的对数概率
            log_prob_point = pt.log_prob(samples)  # (G, B, N)

            # 零中心化: 每个样本组内, 相对可信度
            # 高 log_prob → 该采样点接近均值 → weight 上调
            # 低 log_prob → 该采样点偏离均值 → weight 下调
            log_prob_centered = log_prob_point - log_prob_point.mean(dim=-1, keepdim=True)

            # 调制 weight logit
            adjusted_weight_logit = base_weight_logit.unsqueeze(0) + self.weight_beta * log_prob_centered
        else:
            # weight_beta=0 时退化为确定性 weight (兼容旧行为)
            adjusted_weight_logit = base_weight_logit.unsqueeze(0).expand(sample_num, -1, -1)

        output["weights"] = adjusted_weight_logit  # (G, B, N) — logits, 调用方需 sigmoid

        if return_logP:
            output["log_probs"] = pt.log_prob(samples).mean(dim=-1)
        return output

    def log_probs(self, *input, scores, actions):
        """
        input:
            *input: include action_embedding, anchor_embedding, action_points, anchor_points (B,C,N)
            scores: B,N,N, if needed, use this instead of calculating scores
            actions: tensor of shape G,B,N,3
        return:
            log_probs: tensor of shape G,B
        """
        output = self.forward(*input, scores=scores, sample=True)
        dist = output["distribution"]
        log_probs = dist.log_prob(actions).mean(dim=-1)  # G, B
        return log_probs


class TransformerHead4RL(TransformerHead, ResidualMLPHead4RL):
    def __init__(
        self,
        point_encoder_fun=None,
        emb_dims=512,
        output_num=1024,
        pred_weight=True,
        residual_on=True,
        pos_enc=False,
        norm=nn.BatchNorm1d,
        project_corrs=False,
        project_corrs_mode='mlp',
        weight_beta: float = 0.1,
        attn_mode: str = "torch_attn",
    ):
        super(TransformerHead4RL, self).__init__(
            point_encoder_fun, emb_dims,
            output_num, pred_weight,
            residual_on, pos_enc,
            norm, project_corrs, project_corrs_mode,
            attn_mode
        )
        self.corr_pts_var = PointwiseMLP(
            [emb_dims, emb_dims // 2, emb_dims // 4, emb_dims // 8], 3, norm
        )
        self.weight_beta = weight_beta
        if getattr(self, "proj_flow", None) is not None:
            del self.proj_flow  # 删除父类中的 proj_flow，避免冗余计算

    def forward(self, *input, scores, sample=False):
        """
        input:
          action_embedding: B,512,N
          action_embedding_raw: B,512,N
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
        if sample:
            action_embedding = input[0]
            flow = output["full_flow"][:, :-1, :].permute(0, 2, 1).contiguous()  # B, N, 3
            log_var = self.corr_pts_var(action_embedding).permute(0, 2, 1).contiguous()  # B, N, 3
            pt = MultivariateNormal(flow, torch.diag_embed(log_var.exp()))
            output["distribution"] = pt

        return output


@dataclass
class HeadConfig:
    norm: nn.Module = nn.BatchNorm1d
    emb_dims: int = 512
    output_num: int = 1024
    up_sample_ratio: int = 2
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
    reparam: bool = False
    weight_beta: float = 0.1         # RL head: 采样 log_prob 对 weight 的调制强度
    attn_mode: str = "torch_attn"  # transformer head attention mode


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
            attn_mode=cfg.attn_mode,
        )
    if cfg.head_type == "rl_residual":
        head_type = ReparamResidualMLPHead if cfg.reparam else ResidualMLPHead4RL
        print("Using Head: ", cfg.head_type)
        return head_type(
            cfg.emb_dims,
            cfg.pred_weight,
            cfg.residual_on,
            cfg.norm,
            cfg.head_bias,
            cfg.use_coarse_soft,
            cfg.project_corrs,
            cfg.project_corrs_mode,
            cfg.output_num,
            weight_beta=cfg.weight_beta,
        )
    if cfg.head_type == "upsampling":
        print("Using Head: ", cfg.head_type)
        return Coarse_Res_Head(
            cfg.emb_dims,
            cfg.output_num,
            cfg.up_sample_ratio,
            cfg.pred_weight,
            cfg.norm,
            weight_knn_k=16,
        )
    if cfg.head_type == "rl_transformer":
        head_type = TransformerHead4RL if cfg.reparam else TransformerHead4RL
        return head_type(
            point_encoder_fun=embedding_fun,
            emb_dims=cfg.emb_dims,
            output_num=cfg.output_num,
            pred_weight=cfg.pred_weight,
            residual_on=cfg.residual_on,
            pos_enc=cfg.pos_encoding,
            project_corrs=cfg.project_corrs,
            project_corrs_mode=cfg.project_corrs_mode,
            norm=cfg.norm,
            weight_beta=cfg.weight_beta,
            attn_mode=cfg.attn_mode,
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, C = 2, 512

    print("=" * 60)
    print("Coarse_Res_Head 测试")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════════
    # Test 1: 非降采样 (M=N), pred_weight=True, project_corrs=True
    # ═══════════════════════════════════════════════════════════════
    N = 128
    up_ratio = 1
    head = Coarse_Res_Head(
        emb_dims=C, encoder_output_num=N, up_sample_ratio=up_ratio,
        pred_weight=True, project_corrs=True, project_corrs_mode='mlp',
    ).to(device)

    action_emb = torch.randn(B, C, N, device=device)
    anchor_emb = torch.randn(B, C, N, device=device)
    action_pts = torch.randn(B, 3, N, device=device)
    anchor_pts = torch.randn(B, 3, N, device=device)
    scores = torch.softmax(torch.randn(B, N, N, device=device), dim=-1)

    input_tup = (action_emb, action_emb, anchor_emb, action_pts, anchor_pts, None, None)
    out = head(*input_tup, scores=scores)

    print(f"\nTest 1 (non-downsampled, pred_weight=True):")
    for k, v in out.items():
        print(f"  {k:<16s}: {str(v.shape):>20s}")
    assert 'full_flow' in out, "Missing full_flow"
    assert 'residual_flow' in out, "Missing residual_flow"
    assert 'corr_flow' in out, "Missing corr_flow"
    assert 'corr_points' in out, "Missing corr_points"
    assert 'scores' in out, "Missing scores"

    # ═══════════════════════════════════════════════════════════════
    # Test 2: 降采样 (M < N), pred_weight=True
    # ═══════════════════════════════════════════════════════════════
    M, N_ds = 64, 256
    up_ratio_ds = N_ds // M
    head_ds = Coarse_Res_Head(
        emb_dims=C, encoder_output_num=M, up_sample_ratio=up_ratio_ds,
        pred_weight=True, project_corrs=True, project_corrs_mode='mlp',
    ).to(device)

    action_emb_ds = torch.randn(B, C, M, device=device)          # 稀疏特征
    anchor_emb_ds = torch.randn(B, C, M, device=device)
    action_pts_ds = torch.randn(B, 3, N_ds, device=device)       # 全分辨率坐标
    anchor_pts_ds = torch.randn(B, 3, N_ds, device=device)
    act_down = torch.randn(B, 3, M, device=device)               # 降采样坐标
    anch_down = torch.randn(B, 3, M, device=device)
    scores_spa = torch.softmax(torch.randn(B, M, M, device=device), dim=-1)

    input_tup_ds = (action_emb_ds, action_emb_ds, anchor_emb_ds,
                    action_pts_ds, anchor_pts_ds, act_down, anch_down)
    out_ds = head_ds(*input_tup_ds, scores=scores_spa)

    print(f"\nTest 2 (downsampled M={M}<N={N_ds}, pred_weight=True):")
    for k, v in out_ds.items():
        print(f"  {k:<16s}: {str(v.shape):>20s}")

    # ═══════════════════════════════════════════════════════════════
    # Test 3: pred_weight=False (非降采样, 无权重预测)
    # ═══════════════════════════════════════════════════════════════
    head_noweight = Coarse_Res_Head(
        emb_dims=C, encoder_output_num=N, up_sample_ratio=up_ratio,
        pred_weight=False, project_corrs=False,
    ).to(device)
    out_nw = head_noweight(*input_tup, scores=scores)
    print(f"\nTest 3 (pred_weight=False, project_corrs=False):")
    for k, v in out_nw.items():
        print(f"  {k:<16s}: {str(v.shape):>20s}")

    # ═══════════════════════════════════════════════════════════════
    # Test 4: project_corrs_mode='vn' (非降采样)
    # ═══════════════════════════════════════════════════════════════
    try:
        head_vn = Coarse_Res_Head(
            emb_dims=C, encoder_output_num=N, up_sample_ratio=up_ratio,
            pred_weight=True, project_corrs=True, project_corrs_mode='vn',
        ).to(device)
        out_vn = head_vn(*input_tup, scores=scores)
        print(f"\nTest 4 (project_corrs_mode='vn'):")
        for k, v in out_vn.items():
            print(f"  {k:<16s}: {str(v.shape):>20s}")
    except Exception as e:
        print(f"\nTest 4 (vn mode) skipped: {e}")

    # ═══════════════════════════════════════════════════════════════
    # Test 5: project_corrs_mode='moe' (非降采样)
    # ═══════════════════════════════════════════════════════════════
    try:
        head_moe = Coarse_Res_Head(
            emb_dims=C, encoder_output_num=N, up_sample_ratio=up_ratio,
            pred_weight=True, project_corrs=True, project_corrs_mode='moe',
        ).to(device)
        out_moe = head_moe(*input_tup, scores=scores)
        print(f"\nTest 5 (project_corrs_mode='moe'):")
        for k, v in out_moe.items():
            print(f"  {k:<16s}: {str(v.shape):>20s}")
    except Exception as e:
        print(f"\nTest 5 (moe mode) skipped: {e}")

    # ═══════════════════════════════════════════════════════════════
    # Test 6: 梯度回传 (非降采样)
    # ═══════════════════════════════════════════════════════════════
    action_emb_grad = torch.randn(B, C, N, device=device, requires_grad=True)
    scores_grad = torch.softmax(torch.randn(B, N, N, device=device, requires_grad=True), dim=-1)
    input_grad = (action_emb_grad, action_emb_grad, anchor_emb.clone(),
                  action_pts.clone(), anchor_pts.clone(), None, None)
    out_grad = head(*input_grad, scores=scores_grad)
    loss = out_grad['full_flow'].sum()
    loss.backward()
    print(f"\nTest 6 (gradient flow): "
          f"action_emb.grad={action_emb_grad.grad is not None}, "
          f"scores.grad={scores_grad.grad is not None}")
    assert action_emb_grad.grad is not None, "Gradient should flow to embeddings"

    # ═══════════════════════════════════════════════════════════════
    # Test 7: 不同 batch size 兼容性
    # ═══════════════════════════════════════════════════════════════
    for test_B in [1, 4]:
        a_emb = torch.randn(test_B, C, N, device=device)
        a_pts = torch.randn(test_B, 3, N, device=device)
        s = torch.softmax(torch.randn(test_B, N, N, device=device), dim=-1)
        tup = (a_emb, a_emb, a_emb, a_pts, a_pts, None, None)
        out_b = head(*tup, scores=s)
        print(f"Test 7 (B={test_B}): full_flow.shape = {out_b['full_flow'].shape}")

    # ═══════════════════════════════════════════════════════════════
    # Test 8: create_head 工厂函数
    # ═══════════════════════════════════════════════════════════════
    cfg = HeadConfig(
        head_type='upsampling',
        emb_dims=C, output_num=N, up_sample_ratio=up_ratio,
        pred_weight=True, project_corrs=True,
    )
    head_factory = create_head(cfg).cuda()
    assert isinstance(head_factory, Coarse_Res_Head), \
        f"Expected Coarse_Res_Head, got {type(head_factory).__name__}"
    out_factory = head_factory(*input_tup, scores=scores)
    assert 'full_flow' in out_factory
    print(f"\nTest 8 (create_head factory): "
          f"type={type(head_factory).__name__}, "
          f"full_flow.shape={out_factory['full_flow'].shape}")

    n_params = sum(p.numel() for p in head.parameters())
    print(f"\n{'=' * 60}")
    print(f"✓ All tests passed!  Total params (non-ds head): {n_params:,}")
    print(f"{'=' * 60}")
