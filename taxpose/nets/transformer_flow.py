#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Pulled from DCP

import copy
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, cast

from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F

from taxpose.nets.pointnet import PointwiseMLP, ResidualPointNet, PointNet
from taxpose.nets.transformer_flow_pm import CustomTransformer
from taxpose.nets.tv_mlp import MLP as TVMLP
from taxpose.utils.multilateration import estimate_p
from third_party.dcp.model import DGCNN
from taxpose.nets.raw_dgcnn import DGCNN4TaxPose, DGCNN_VAE, knn
from taxpose.nets.vn_dgcnn import VN_DGCNN_iqSO3, VNArgs
from taxpose.nets.dgcnn_group_v2 import DGCNN_Grouper_V2
from taxpose.nets.head import create_head, HeadConfig
from taxpose.nets.gemo_fea import ManualPointWiseGemoFea
from taxpose.nets.huggingface_tf import Transformer
from taxpose.nets.head import TransformerHead
from taxpose.utils.se3 import dualflow2pose, flow2pose
from pytorch3d.transforms import (
    Rotate,
    Transform3d,
    Translate,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    so3_rotation_angle,
    so3_relative_angle
)
from pytorch3d.ops import knn_points, knn_gather, estimate_pointcloud_normals


def knn_points_with_normals(
    p1: torch.Tensor,
    p2: torch.Tensor,
    n1: torch.Tensor = None,
    n2: torch.Tensor = None,
    K: int = 1,
    normal_weight: float = 0.5,
    return_nn: bool = False,
    return_sorted: bool = True,
):
    """融合位置和法向量信息的 KNN 搜索。

    将位置和加权法向量拼接为 6D 特征，利用标准 L2 KNN 实现
    联合距离度量:
        dist² = ||p₁-p₂||² + λ²·||n₁-n₂||²

    法向量差 L2 与余弦相似度的关系:
        ||n₁-n₂||² = 2 - 2·cos(θ)
        对齐时(cos=1) → 0，相反时(cos=-1) → 4

    Args:
        p1, p2: (B, N, 3)  位置坐标
        n1, n2: (B, N, 3)  法向量（自动归一化）
        K: 近邻数量
        normal_weight: 法向量权重 λ，越大法向一致性越重要
        return_nn: 是否返回最近邻的点
        return_sorted: 是否按距离排序
    Returns:
        idx:   (B, N, K)  近邻索引
        nn:    (B, N, K, 6)  近邻的 [位置, λ·法向量]（仅 return_nn=True）
    """
    if n1 is None or n2 is None:
        return knn_points(
            p1, p2, K=K,
            return_nn=return_nn,
            return_sorted=return_sorted,
        )

    # 归一化法向量
    n1 = F.normalize(n1, dim=-1)
    n2 = F.normalize(n2, dim=-1)

    cdist = torch.cdist(p1, p2)
    abs_cos_sim = torch.abs(torch.matmul(n1, n2.transpose(-2, -1)))  # (B, N, M)
    dist = cdist + normal_weight * (1 - abs_cos_sim)
    
    idx = torch.topk(dist, k=K, dim=-1, largest=False, sorted=return_sorted)[1]

    if return_nn:
        nn = knn_gather(p2, idx)
        return idx, nn
    return idx, None


def knn_query_with_embedding(embedding_q, embedding_k,
                             K: int = 1, return_sorted: bool = True):
    """基于 embedding 余弦相似度的 KNN 查询。

    通过 L2 归一化 + matmul 计算全部 pairwise 余弦相似度，
    然后取 top-K 索引。

    Args:
        embedding_q: (B, N, C) — query embedding（特征必须在最后一维）
        embedding_k: (B, M, C) — key embedding
        K: 近邻数
    Returns:
        idx: (B, N, K) — 每个 query 点在 key 中的 K 近邻索引
    """
    emb_q = F.normalize(embedding_q, dim=-1)
    emb_k = F.normalize(embedding_k, dim=-1)
    # (B, N, C) @ (B, C, M) → (B, N, M), 每行是 query 点对所有 key 点的相似度
    sim = torch.matmul(emb_q, emb_k.transpose(-2, -1))
    idx = torch.topk(sim, k=K, dim=-1, sorted=return_sorted)[1]
    return idx


class EquivariantFeatureEmbeddingNetwork(nn.Module):
    def __init__(self, encoder_cfg):
        super(EquivariantFeatureEmbeddingNetwork, self).__init__()
        self.emb_nn = create_embedding_network(encoder_cfg)

    def forward(self, *input):
        points = input[0]  # B, 3, num_points
        points_dmean = points - points.mean(dim=2, keepdim=True)

        points_embedding = self.emb_nn(points_dmean)  # B, emb_dims, num_points
        if isinstance(points_embedding, tuple):
            points_embedding, points = points_embedding
        return points_embedding, points.transpose(1, 2).contiguous()

    def get_group_centers(self):
        return self.emb_nn.coord.transpose(1, 2).contiguous().detach()


class CorrespondenceFlow_DiffEmbMLP(nn.Module):
    def __init__(self, encoder_cfg, cycle=True, center_feature=True):
        super(CorrespondenceFlow_DiffEmbMLP, self).__init__()
        self.cycle = cycle

        self.emb_nn_action = create_embedding_network(encoder_cfg)
        self.emb_nn_anchor = create_embedding_network(encoder_cfg)
        emb_dims = encoder_cfg.encoder

        self.center_feature = center_feature

        self.transformer_action = MLP(emb_dims=emb_dims)
        self.transformer_anchor = MLP(emb_dims=emb_dims)
        self.head_action = CorrespondenceMLPHead(emb_dims=emb_dims)
        self.head_anchor = CorrespondenceMLPHead(emb_dims=emb_dims)

    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)[:, :3]  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)[:, :3]
        action_points_dmean = action_points - action_points.mean(dim=2, keepdim=True)
        anchor_points_dmean = anchor_points - anchor_points.mean(dim=2, keepdim=True)
        # mean center point cloud before DGCNN
        if not self.center_feature:
            action_points_dmean = action_points
            anchor_points_dmean = anchor_points
        action_embedding = self.emb_nn_action(action_points_dmean)
        anchor_embedding = self.emb_nn_anchor(anchor_points_dmean)

        # tilde_phi, phi are both B,512,N
        action_embedding_tf = self.transformer_action(action_embedding)
        # action_embedding_tf: Batch, emb_dim, num_points
        # action_attn: Batch, 4, num_points, num_points
        anchor_embedding_tf = self.transformer_anchor(anchor_embedding)
        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf

        flow_action = self.head_action(
            action_embedding_tf,
            anchor_embedding_tf,
            action_points,
            anchor_points,
            scores=None,
        ).permute(0, 2, 1)

        outputs = {
            "flow_action": flow_action,
        }

        if self.cycle:
            flow_anchor = self.head_anchor(
                anchor_embedding_tf,
                action_embedding_tf,
                anchor_points,
                action_points,
                scores=None,
            ).permute(0, 2, 1)
            outputs["flow_anchor"] = flow_anchor

        return outputs


class CorrespondenceMLPHead(nn.Module):
    """
    Output correspondence flow and weight
    """

    def __init__(self, emb_dims=512):
        super(CorrespondenceMLPHead, self).__init__()

        self.emb_dims = emb_dims
        self.proj_flow = nn.Sequential(
            PointNet([emb_dims, 64, 64, 64, 128, 512]),
            # PointNet([emb_dims, emb_dims//2, emb_dims//4, emb_dims//8]),
            nn.Conv1d(512, 1, kernel_size=1, bias=False),
        )

    def forward(self, *input, scores=None):
        action_embedding = input[0]
        anchor_embedding = input[1]
        action_points = input[2]
        anchor_points = input[3]
        if scores is None:
            if len(input) <= 4:
                action_query = action_embedding
                anchor_key = anchor_embedding
            else:
                action_query = input[4]
                anchor_key = input[5]

            d_k = action_query.size(1)
            scores = torch.matmul(
                action_query.transpose(2, 1), anchor_key
            ) / math.sqrt(d_k)
            # W_i # B, N, N (N=number of points, 1024 cur)
            scores = torch.softmax(scores, dim=2)

        corr_points = torch.matmul(anchor_points, scores.transpose(2, 1))
        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points
        weight = self.proj_flow(action_embedding)
        corr_flow_weight = torch.concat([corr_flow, weight], dim=1)

        return corr_flow_weight


class MLP(nn.Module):
    def __init__(self, emb_dims=512):
        super(MLP, self).__init__()
        self.input_fc = nn.Linear(emb_dims, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, emb_dims)

    def forward(self, x):
        # x = [batch size, emb_dims, num_points]
        batch_size, _, num_points = x.shape
        x = x.permute(0, -1, -2)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        h_1 = F.relu(self.input_fc(x))
        # batch size*num_points, 100
        h_2 = F.relu(self.hidden_fc(h_1))

        # batch size*num_points, output dim
        y_pred = self.output_fc(h_2)
        # batch size, num_points, output dim
        y_pred = y_pred.view(batch_size, num_points, -1)
        # batch size, emb_dims, num_points
        y_pred = y_pred.permute(0, 2, 1)

        return y_pred


class MLPHead(nn.Module):
    def __init__(self, emb_dims=512):
        super(MLPHead, self).__init__()

        self.emb_dims = emb_dims
        self.proj_flow = nn.Sequential(
            PointNet([emb_dims, emb_dims // 2, emb_dims // 4, emb_dims // 8]),
            nn.Conv1d(emb_dims // 8, 3, kernel_size=1, bias=False),
        )

    def forward(self, *input):
        action_embedding = input[0]
        embedding = action_embedding
        flow = self.proj_flow(embedding)
        return flow


class MLPHeadWeight(nn.Module):
    def __init__(self, emb_dims=512):
        super(MLPHeadWeight, self).__init__()

        self.emb_dims = emb_dims
        self.proj_flow = nn.Sequential(
            PointNet([emb_dims, emb_dims // 2, emb_dims // 4, emb_dims // 8]),
            nn.Conv1d(emb_dims // 8, 4, kernel_size=1, bias=False),
        )

    def forward(self, *input):
        action_embedding = input[0]
        embedding = action_embedding
        flow = self.proj_flow(embedding)
        return flow


class MLPKernel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = TVMLP(2 * feature_dim, [300, 100, 1])

    def forward(self, x1, x2):
        # Make it symmetric.
        # b = torch.stack(
        #     [
        #         torch.cat([x1, x2], axis=-1),
        #         torch.cat([x2, x1], axis=-1),
        #     ],
        #     axis=0,
        # )
        v1 = self.mlp(torch.cat([x1, x2], axis=-1))
        v2 = self.mlp(torch.cat([x2, x1], axis=-1))
        return F.softplus((v1 + v2) / 2)


class NormKernel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x1, x2):
        return torch.norm(x1 - x2, dim=-1) / math.sqrt(len(x1))


class DotProductKernel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x1, x2):
        return torch.dot(x1, x2) / math.sqrt(len(x1))


class MultilaterationHead(nn.Module):
    def __init__(
        self,
        emb_dims=512,
        n_kps=100,
        pred_weight=True,
        last_attn=False,
        sample: bool = False,
    ):
        super().__init__()

        self.emb_dims = emb_dims
        self.n_kps = n_kps
        self.last_attn = last_attn

        self.kernel = MLPKernel(self.emb_dims - int(last_attn))
        # self.kernel = NormKernel(self.emb_dims)
        self.sample = sample

        self.pred_weight = pred_weight
        if self.pred_weight:
            self.proj_flow_weight = nn.Sequential(
                PointNet([emb_dims - int(last_attn), 64, 64, 64, 128, 512]),
                nn.Conv1d(512, 1, kernel_size=1, bias=False),
            )

    def forward(
        self, *input, scores=None, return_flow_component=False, return_embedding=False
    ):
        action_embedding = input[0]
        anchor_embedding = input[1]

        if self.last_attn:
            action_embedding, action_attn = (
                action_embedding[:, :-1],
                action_embedding[:, -1:],
            )
            anchor_embedding, anchor_attn = (
                anchor_embedding[:, :-1],
                anchor_embedding[:, -1:],
            )

        action_points = input[2]
        anchor_points = input[3]

        P_A = action_points.permute(0, 2, 1)
        P_B = anchor_points.permute(0, 2, 1)

        Phi_A = action_embedding.permute(0, 2, 1)
        Phi_B = anchor_embedding.permute(0, 2, 1)

        if self.last_attn:
            A_weights = action_attn.permute(0, 2, 1)
            B_weights = anchor_attn.permute(0, 2, 1)

            A_weights = F.softmax(A_weights, dim=-1).squeeze(dim=-1)
            B_weights = F.softmax(B_weights, dim=-1).squeeze(dim=-1)

            # Should sum to N.
            A_weights = A_weights * A_weights.shape[-1]
            B_weights = B_weights * B_weights.shape[-1]
        else:
            A_weights = torch.ones(Phi_A.shape[:2], device=Phi_A.device)
            B_weights = torch.ones(Phi_B.shape[:2], device=Phi_B.device)

        # We probably want to sample
        if self.sample:
            # This function samples without replacement, in a batch.
            choice_v = torch.vmap(
                lambda x, n: torch.randperm(x.shape[-1])[:n],
                in_dims=(0, None),
                randomness="different",
            )
            A_ixs = choice_v(action_points, self.n_kps).to(action_points.device)
            B_ixs = choice_v(anchor_points, self.n_kps).to(anchor_points.device)
            P_A = torch.take_along_dim(P_A, A_ixs.unsqueeze(-1), dim=1)
            Phi_A = torch.take_along_dim(Phi_A, A_ixs.unsqueeze(-1), dim=1)
            P_B = torch.take_along_dim(P_B, B_ixs.unsqueeze(-1), dim=1)
            Phi_B = torch.take_along_dim(Phi_B, B_ixs.unsqueeze(-1), dim=1)
            A_weights = torch.take_along_dim(A_weights, A_ixs, dim=1)
            B_weights = torch.take_along_dim(B_weights, B_ixs, dim=1)
        else:
            bs = P_A.shape[0]
            A_ixs = torch.arange(P_A.shape[1], device=P_A.device).repeat(bs, 1)
            B_ixs = torch.arange(P_B.shape[1], device=P_B.device).repeat(bs, 1)

        # compute_R = torch.vmap(
        #     torch.vmap(
        #         torch.vmap(self.kernel, in_dims=(None, 0)), in_dims=(0, None)
        #     ),
        #     in_dims=(0, 0),
        # )
        # R_est = compute_R(Phi_A, Phi_B)
        Phi_A_r = (
            Phi_A.unsqueeze(2)
            .repeat(1, 1, Phi_A.shape[1], 1)
            .reshape(Phi_A.shape[0] * Phi_A.shape[1] * Phi_A.shape[1], Phi_A.shape[2])
        )
        Phi_B_r = (
            Phi_B.unsqueeze(1)
            .repeat(1, Phi_B.shape[1], 1, 1)
            .reshape(Phi_B.shape[0] * Phi_B.shape[1] * Phi_B.shape[1], Phi_B.shape[2])
        )
        R_est = self.kernel(Phi_A_r, Phi_B_r).reshape(
            Phi_A.shape[0], Phi_A.shape[1], Phi_B.shape[1]
        )

        # R_est = torch.cdist(Phi_A, Phi_B, p=2.0) / math.sqrt(self.emb_dims)

        # Normalize the scores.
        # mlat_weights = (
        #     scores / scores.detach().sum(dim=-1, keepdim=True) * scores.shape[-1]
        # )
        # mlat_weights = torch.ones_like(scores, device=scores.device)
        v_est_p = torch.vmap(torch.vmap(estimate_p, in_dims=(None, 0, None)))
        P_A_B_pred = v_est_p(P_B[..., None], R_est, B_weights)[..., 0]

        corr_points = P_A_B_pred.permute(0, 2, 1)
        flow = (P_A_B_pred - P_A).permute(0, 2, 1)

        # TODO: figure out how to downsample the points, and pass it all back up the stack.

        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        # flow = corr_points - action_points

        if self.pred_weight:
            weight = self.proj_flow_weight(action_embedding)
            if self.sample:
                weight = torch.take_along_dim(weight, A_ixs.unsqueeze(1), dim=2)
            corr_flow_weight = torch.concat([flow, weight], dim=1)
        else:
            corr_flow_weight = flow

        return {
            "full_flow": corr_flow_weight,
            "residual_flow": torch.zeros_like(flow).to(flow.device),
            "corr_flow": flow,
            "corr_points": corr_points,
            "scores": scores,
            "P_A": P_A.permute(0, 2, 1),
            "A_ixs": A_ixs,
        }


def create_embedding_network(cfg) -> nn.Module:
    if cfg.name == "dgcnn":
        network: nn.Module = DGCNN(emb_dims=cfg.emb_dims)
    elif cfg.name == "vn_dgcnn":
        print(f"Using {cfg.name} with iqSO3 pooling")
        network = VN_DGCNN_iqSO3(
            emb_dims=cfg.emb_dims,
            knn=cfg.knn,
            down_sample=cfg.down_ratio > 1,
            down_ratio=cfg.down_ratio,
            output_num=cfg.output_num,
            pos_encoding=cfg.pos_encoding,
            norm_mode=cfg.norm,
            pooling=cfg.pooling,
        )
    elif cfg.name == "raw_dgcnn":
        print(f"Using {cfg.name}")
        network: nn.Module = DGCNN4TaxPose(
            cfg.emb_dims, cfg.knn,
            cfg.dropout, cfg.norm,
            output_c=cfg.output_num)
    elif cfg.name == "dgcnn_group":
        print(f"Using {cfg.name} with grouping")
        network = DGCNN_Grouper_V2(
            cfg.emb_dims, cfg.output_num, cfg.knn, cfg.dropout,
            cfg.norm, downsample_layers=cfg.down_layers)
    elif cfg.name == "vae_dgcnn":
        print(f"Using {cfg.name}")
        network = DGCNN_VAE(
            cfg,
            pos_encoding=cfg.pos_encoding,
        )
    else:
        raise ValueError(f"Unknown embedding network type: {cfg.name}")

    return network


class ResidualFlow_DiffEmbTransformer(nn.Module):
    head_norm = {
        "BN": nn.BatchNorm1d,
        "LN": nn.LayerNorm,
        "IN": nn.InstanceNorm1d
    }

    def __init__(
        self,
        encoder_cfg,
        head_cfg,
        cycle=True,
        center_feature=False,
        freeze_embnn=False,
        return_attn=True,
        multilaterate=False,
        mlat_sample: bool = False,
        mlat_nkps: int = 100,
        feature_channels=0,  # Number of extra channels we'll pass into the network.
        conditional: bool = False,
        dropout=0.1,
        pos_encoding=False,
        n_blocks=1,
        attn_mode="torch_attn",
        fine_tune=False,
        weight_beta=0.1,
        stage=0,
        is_final=True,
        **kwargs
    ):
        super(ResidualFlow_DiffEmbTransformer, self).__init__()
        self.cycle = cycle
        self.feature_channels = feature_channels

        self.emb_nn_action = create_embedding_network(encoder_cfg)
        self.emb_nn_anchor = create_embedding_network(encoder_cfg) if freeze_embnn else self.emb_nn_action
        emb_dims = encoder_cfg.emb_dims
        self.freeze_embnn = freeze_embnn
        if freeze_embnn:
            self.emb_nn_action.requires_grad_(False)
            self.emb_nn_anchor.requires_grad_(False)
        self.center_feature = center_feature
        self.return_attn = return_attn
        self.conditional = conditional

        self.transformer_action = CustomTransformer(
            emb_dims=emb_dims,
            n_blocks=n_blocks,
            dropout=dropout,
            ff_dims=4*emb_dims,
            n_heads=emb_dims//64,
            return_attn=self.return_attn,
            bidirectional=False,
            attn_mode=attn_mode
        )
        self.transformer_anchor = CustomTransformer(
            emb_dims=emb_dims,
            n_blocks=n_blocks,
            dropout=dropout,
            ff_dims=4*emb_dims,
            n_heads=emb_dims//64,
            return_attn=self.return_attn,
            bidirectional=False,
            attn_mode=attn_mode
        )
        if multilaterate:
            self.head_action = MultilaterationHead(
                emb_dims=emb_dims,
                pred_weight=self.pred_weight,
                sample=mlat_sample,
                n_kps=mlat_nkps,
            )
            self.head_anchor = MultilaterationHead(
                emb_dims=emb_dims,
                pred_weight=self.pred_weight,
                sample=mlat_sample,
                n_kps=mlat_nkps,
            )
        else:
            if not isinstance(head_cfg, HeadConfig):
                norm = self.head_norm.get(head_cfg.norm, nn.LayerNorm)
                head_cfg = OmegaConf.to_container(head_cfg, resolve=True)
                head_cfg.pop("norm")
                cfg = HeadConfig(
                    **head_cfg,
                    output_num=encoder_cfg.output_num,
                    norm=norm, emb_dims=emb_dims, pos_encoding=pos_encoding)
            else:
                cfg = head_cfg
            self.head_action: nn.Module = create_head(
                cfg, embedding_fun=self._action_embedding
            )
            self.head_anchor: nn.Module = create_head(
                cfg, embedding_fun=self._anchor_embedding
            )

        if self.conditional:
            # Simple projection to the embedding space. This will be concatenated to the embeddings at
            # the attention layer.
            self.proj_onehot = nn.Linear(5, emb_dims)

        if self.feature_channels > 0:
            # We're basically putting a few MLP layers in on top of the invariant module.
            combined_dims = emb_dims + self.feature_channels
            self.feature_channel_encoder_action = nn.Sequential(
                PointNet([combined_dims, combined_dims * 2, combined_dims * 4]),
                nn.Conv1d(combined_dims * 4, emb_dims, kernel_size=1, bias=False),
            )
            self.feature_channel_encoder_anchor = nn.Sequential(
                PointNet([combined_dims, combined_dims * 2, combined_dims * 4]),
                nn.Conv1d(combined_dims * 4, emb_dims, kernel_size=1, bias=False),
            )

        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pos_encoder = ManualPointWiseGemoFea(True, emb_dims)

        self.fine_tune = fine_tune
        self.weight_beta = weight_beta
        self.stage = stage
        self.is_final_stage = is_final
        self.knn_in_tf = "knn" in attn_mode

    def get_parameters(self, module: str):
        """返回指定模块的参数列表，用于分层学习率。

        Args:
            module: 模块名，可选 "emb", "backbone", "head", "gru"
        Returns:
            参数迭代器或空列表
        """
        if module == "emb":
            params = list(self.emb_nn_action.parameters()) + \
                     list(self.emb_nn_anchor.parameters())
            if self.pos_encoding:
                params += list(self.pos_encoder.parameters())
            if self.feature_channels > 0:
                params += list(self.feature_channel_encoder_action.parameters())
                params += list(self.feature_channel_encoder_anchor.parameters())
            return params
        elif module == "backbone":
            params = list(self.transformer_action.parameters()) + \
                     list(self.transformer_anchor.parameters())
            if self.conditional:
                params += list(self.proj_onehot.parameters())
            return params
        elif module == "head":
            params = list(self.head_action.parameters())
            if self.cycle:
                params += list(self.head_anchor.parameters())
            return params
        elif module == "gru":
            return []  # 父类无 GRU
        else:
            raise ValueError(f"Unknown module: {module}. "
                             f"Expected 'emb', 'backbone', 'head', or 'gru'.")

    def _action_embedding(self, points, allow_down=False):
        """
        Embedding function for the action point cloud.
        Args:
            points: B,3,num_points
            allow_down: Allow downsampling.
        """
        self.emb_nn_action.train(not self.freeze_embnn)
        points = points - points.mean(dim=2, keepdim=True)
        embedding = self.emb_nn_action(points, down=allow_down)
        # embedding = F.normalize(embedding, dim=1)
        return embedding

    def _anchor_embedding(self, points, allow_down=False):
        self.emb_nn_action.train(not self.freeze_embnn)
        points = points - points.mean(dim=2, keepdim=True)
        embedding = self.emb_nn_anchor(points, down=allow_down)
        # embedding = F.normalize(embedding, dim=1)
        return embedding

    def _embedding(self, *input):
        action_points = input[0].permute(0, 2, 1).contiguous()[:, :3]  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1).contiguous()[:, :3]
        action_points_dmean = action_points - action_points.mean(dim=2, keepdim=True)
        anchor_points_dmean = anchor_points - anchor_points.mean(dim=2, keepdim=True)
        # mean center point cloud before DGCNN
        if not self.center_feature:
            action_points_dmean = action_points
            anchor_points_dmean = anchor_points
        if self.freeze_embnn:
            self.emb_nn_action.eval()
        with torch.set_grad_enabled(not self.freeze_embnn):
            act_down_sample, anch_down_sample = None, None
            # with torch.set_grad_enabled(not self.freeze_embnn):
            action_embedding = self.emb_nn_action(action_points_dmean)
            if isinstance(action_embedding, tuple):
                action_embedding, pts = action_embedding
                act_down_sample = pts + action_points.mean(dim=2, keepdim=True)
                # action_points = pts + action_points.mean(dim=2, keepdim=True)
            anchor_embedding = self.emb_nn_anchor(anchor_points_dmean)
            if isinstance(anchor_embedding, tuple):
                anchor_embedding, pts = anchor_embedding
                anch_down_sample = pts + anchor_points.mean(dim=2, keepdim=True)
                # anchor_points = pts + anchor_points.mean(dim=2, keepdim=True)
            # action_embedding = F.normalize(action_embedding, dim=1)
            # anchor_embedding = F.normalize(anchor_embedding, dim=1)
            del action_points_dmean, anchor_points_dmean
            if self.feature_channels > 0:
                # Add a symmetry label to the embeddings.
                action_features = input[2].permute(0, 2, 1).contiguous()
                anchor_features = input[3].permute(0, 2, 1).contiguous()

                action_embedding_stack = torch.cat(
                    [action_embedding, action_features], axis=1
                )
                anchor_embedding_stack = torch.cat(
                    [anchor_embedding, anchor_features], axis=1
                )

                action_embedding = self.feature_channel_encoder_action(
                    action_embedding_stack
                )

                anchor_embedding = self.feature_channel_encoder_anchor(
                    anchor_embedding_stack
                )

            if self.conditional:
                # We first project the one-hot encoding to the embedding space.
                onehot = input[4].float()  # B x C
                # Extend the onehot vector so that C becomes 5.
                onehot = F.pad(onehot, (0, 5 - onehot.shape[-1]), "constant", 0)
                onehot_emb = self.proj_onehot(onehot)

                # Then, we do a linear addition to the embeddings. This should broadcast correctly.
                action_embedding = action_embedding + onehot_emb[..., None]
                anchor_embedding = anchor_embedding + onehot_emb[..., None]

            # tilde_phi, phi are both B,512,N
            # Get the new cross-attention embeddings.
            action_pt_pos, anchor_pt_pos = None, None
            if self.pos_encoding:
                action_pt_pos = self.pos_encoder(action_points)  # B,C,N
                anchor_pt_pos = self.pos_encoder(anchor_points)
                action_embedding += action_pt_pos
                anchor_embedding += anchor_pt_pos
            return (
                action_points, anchor_points,
                action_embedding, anchor_embedding,
                action_pt_pos, anchor_pt_pos,
                act_down_sample, anch_down_sample,
            )

    def _backbone(self, action_embedding, anchor_embedding,
                  action_pt_pos, anchor_pt_pos,
                  act_index=None, anch_index=None):
        action_embedding = action_embedding.transpose(2, 1).contiguous()
        anchor_embedding = anchor_embedding.transpose(2, 1).contiguous()

        transformer_action_outputs = self.transformer_action(
            action_embedding, anchor_embedding, act_index
        )
        transformer_anchor_outputs = self.transformer_anchor(
            anchor_embedding, action_embedding, anch_index
        )
        action_embedding_tf = transformer_action_outputs["src_embedding"]
        action_attn = transformer_action_outputs["src_attn"]
        anchor_embedding_tf = transformer_anchor_outputs["src_embedding"]
        anchor_attn = transformer_anchor_outputs["src_attn"]

        if not self.return_attn:
            action_attn = None
            anchor_attn = None
        else:
            action_attn = action_attn.mean(dim=1)  # b, h，N, M -> b, N, M
            anchor_attn = anchor_attn.mean(dim=1)

        if self.pos_encoding:
            action_embedding_tf += action_pt_pos
            anchor_embedding_tf += anchor_pt_pos

        del transformer_action_outputs, transformer_anchor_outputs
        return action_embedding_tf, anchor_embedding_tf, action_attn, anchor_attn

    def coarse_step(
            self,
            action_embedding_tf,
            anchor_embedding_tf,
            action_embedding,
            anchor_embedding,
            action_points,
            anchor_points,
            act_down_sample,
            anch_down_sample,
            action_attn,
            anchor_attn
            ):
        """"
        Coarse step of the network.
        Inputs are with shape of B, C, N
        """
        head_action_output = self.head_action(
            action_embedding_tf,
            action_embedding,
            anchor_embedding,
            action_points,
            anchor_points,
            act_down_sample,
            anch_down_sample,
            scores=action_attn,
        )
        raw_pts = action_points if act_down_sample is None else act_down_sample
        to_next_stage_corr_points = raw_pts + head_action_output["residual_flow"] + head_action_output["corr_flow"]
        flow_action = head_action_output["full_flow"].permute(0, 2, 1).contiguous()
        residual_flow_action = head_action_output["residual_flow"].permute(0, 2, 1).contiguous()
        corr_flow_action = head_action_output["corr_flow"].permute(0, 2, 1).contiguous()
        corr_points_action = head_action_output["corr_points"].permute(0, 2, 1).contiguous()
        corr_std = head_action_output.get("corr_std", None)
        outputs = {
            "flow_action": flow_action,
            "residual_flow_action": residual_flow_action,
            "corr_flow_action": corr_flow_action,
            "corr_points_action": corr_points_action,
            "act_down_sample": (
                None if act_down_sample is None else act_down_sample.permute(0, 2, 1).contiguous()
            ),
            "b3n_act_corr_points": to_next_stage_corr_points
        }
        if corr_std is not None:
            outputs["corr_std_act"] = corr_std.permute(0, 2, 1).contiguous()
        
        #  *************************DEBUG*************************
        outputs.update(attns=action_attn)

        if "P_A" in head_action_output:
            original_points_action = head_action_output["P_A"].permute(0, 2, 1).contiguous()
            outputs["original_points_action"] = original_points_action
            outputs["sampled_ixs_action"] = head_action_output["A_ixs"]

        del head_action_output

        if self.cycle:
            head_anchor_output = self.head_anchor(
                anchor_embedding_tf,
                anchor_embedding,
                action_embedding,
                anchor_points,
                action_points,
                anch_down_sample,
                act_down_sample,
                scores=anchor_attn,
            )
            raw_pts = anchor_points if anch_down_sample is None else anch_down_sample
            to_next_anch_corr_points = raw_pts + head_anchor_output["residual_flow"] + head_anchor_output["corr_flow"]
            flow_anchor = head_anchor_output["full_flow"].permute(0, 2, 1).contiguous()
            residual_flow_anchor = head_anchor_output["residual_flow"].permute(0, 2, 1).contiguous()
            corr_flow_anchor = head_anchor_output["corr_flow"].permute(0, 2, 1).contiguous()
            corr_points_anchor = head_anchor_output["corr_points"].permute(0, 2, 1).contiguous()
            corr_std = head_anchor_output.get("corr_std", None)
            outputs = {
                **outputs,
                "flow_anchor": flow_anchor,
                "residual_flow_anchor": residual_flow_anchor,
                "corr_flow_anchor": corr_flow_anchor,
                "corr_points_anchor": corr_points_anchor,
                "anch_down_sample": (
                    None
                    if anch_down_sample is None
                    else anch_down_sample.permute(0, 2, 1).contiguous()
                ),
                "b3n_anch_corr_points": to_next_anch_corr_points
            }
            if corr_std is not None:
                outputs["corr_std_anch"] = corr_std.permute(0, 2, 1).contiguous()
            if "P_A" in head_anchor_output:
                original_points_anchor = head_anchor_output["P_A"].permute(0, 2, 1).contiguous()
                outputs["original_points_anchor"] = original_points_anchor
                outputs["sampled_ixs_anchor"] = head_anchor_output["A_ixs"]

            del head_anchor_output
        return outputs

    def fine_step(
            self,
            action_points, anchor_points,
            action_attn, anchor_attn,
            action_embedding_tf,
            anchor_embedding_tf,
            action_embedding,
            anchor_embedding,
            act_down_sample,
            anch_down_sample,
            flow_action,
            flow_anchor,
            ):
        """精调步骤
        输入：
        action_points: B, N, 3
        anchor_points: B, M, 3
        action_attn: B, N, M
        anchor_attn: B, N, M
        flow_action: B, N, 4
        flow_anchor: B, M, 4
        """
        pred_flow_action, pred_w_action = flow_action[:, :, :3], flow_action[:, :, 3]
        pred_flow_anchor, pred_w_anchor = flow_anchor[:, :, :3], flow_anchor[:, :, 3]
        action_bn3 = action_points.transpose(1, 2)
        anchor_bn3 = anchor_points.transpose(1, 2)
        coarse_trans = dualflow2pose(
                xyz_src=action_bn3,
                xyz_tgt=anchor_bn3,
                flow_src=pred_flow_action,
                flow_tgt=pred_flow_anchor,
                weights_src=torch.sigmoid(pred_w_action),
                weights_tgt=torch.sigmoid(pred_w_anchor),
                return_transform3d=True,
                normalization_scehme='l1',
                training=True
        )
        points_pred = coarse_trans.transform_points(action_bn3)
        flow_action_err = torch.norm(
            points_pred - action_bn3 - pred_flow_action, dim=-1)

        # Anchor: T^{-1}(q) ≈ q + pred_flow → q ≈ T(q + pred_flow)
        anchor_rigid = coarse_trans.transform_points(
            anchor_bn3 + pred_flow_anchor)
        flow_anchor_err = (
            anchor_bn3 - anchor_rigid).norm(p=2, dim=-1)          # (B, M)
        
        # ── 4. 注意力重加权: 高误差 → 低权重 ──
        beta = getattr(self, 'weight_beta', 0.5)
        adj_action = (1.0 - beta * torch.sigmoid(flow_action_err)
                      ).clamp(min=0.1)                               # (B, N)
        adj_anchor = (1.0 - beta * torch.sigmoid(flow_anchor_err)
                      ).clamp(min=0.1)                               # (B, M)
        # action_attn (B,N,M): 每行乘 action 调整因子 → (B,N,M)*(B,N,1)
        action_attn = action_attn * adj_action.unsqueeze(-1)
        # anchor_attn (B,N,M): 每列乘 anchor 调整因子 → (B,N,M)*(B,1,M)
        anchor_attn = anchor_attn * adj_anchor.unsqueeze(1)
        # ── 5. 重跑 head (修正后的 attention) ──
        refined = self.coarse_step(
            action_embedding_tf, anchor_embedding_tf,
            action_embedding, anchor_embedding,
            action_points, anchor_points,
            act_down_sample, anch_down_sample,
            action_attn, anchor_attn,
        )
        return refined

    def forward(self, *input, **kwargs):
        if self.stage == 0:
            (
                action_points, anchor_points,
                shared_act_embedding, shared_anch_embedding,
                action_pt_pos, anchor_pt_pos,
                act_down_sample, anch_down_sample
            ) = self._embedding(*input)
            action_embedding = shared_act_embedding
            anchor_embedding = shared_anch_embedding
        else:
            (
                action_points, anchor_points,
                shared_act_embedding, shared_anch_embedding,
                action_pt_pos, anchor_pt_pos,
                act_down_sample, anch_down_sample,
                action_cps, anchor_cps   #  B, 3, N
            ) = input
            action_cp_embedding = self._anchor_embedding(action_cps, allow_down=True)
            anchor_cp_embedding = self._anchor_embedding(anchor_cps, allow_down=True)
            action_embedding = action_cp_embedding + shared_act_embedding
            anchor_embedding = anchor_cp_embedding + shared_anch_embedding

        tf_act_proxys = shared_act_embedding
        tf_anch_proxys = shared_anch_embedding  # TODO: 测试使用emb特征KNN而非点云

        act_index = knn(tf_act_proxys, 8) if self.knn_in_tf else None
        anch_index = knn(tf_anch_proxys, 8) if self.knn_in_tf else None
        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn
        ) = self._backbone(action_embedding, anchor_embedding,
                           action_pt_pos, anchor_pt_pos, act_index, anch_index)

        outputs = self.coarse_step(
            action_embedding_tf,
            anchor_embedding_tf,
            action_embedding,
            anchor_embedding,
            action_points,
            anchor_points,
            act_down_sample,
            anch_down_sample,
            action_attn,
            anchor_attn
        )
        if self.fine_tune:
            compute_loss = kwargs.get("compute_loss", None)
            if compute_loss is not None:
                coarse_loss, _ = compute_loss(outputs)
                outputs.update(coarse_loss=coarse_loss)
            outputs.update(self.fine_step(
                action_points,
                anchor_points,
                action_attn,
                anchor_attn,
                action_embedding_tf,
                anchor_embedding_tf,
                action_embedding,
                anchor_embedding,
                act_down_sample,
                anch_down_sample,
                outputs["flow_action"].detach(),
                outputs["flow_anchor"].detach(),
            ))
        action_cps = outputs.pop("b3n_act_corr_points")
        anchor_cps = outputs.pop("b3n_anch_corr_points", None)
        outputs.update(shared_args=[
            action_points, anchor_points,
            shared_act_embedding, shared_anch_embedding,
            action_embedding_tf, anchor_embedding_tf,
            action_pt_pos, anchor_pt_pos,
            act_down_sample, anch_down_sample,
            action_cps, anchor_cps
        ])
        return outputs

    @staticmethod
    def _solve_transformation(action_bn3, flow_act, has_anchor=True,
                              anchor_bn3=None, flow_anchor=None):
        flow_a = flow_act[:, :, :3]
        pw_a = torch.sigmoid(flow_act[:, :, 3])
        if has_anchor:
            assert anchor_bn3 is not None and flow_anchor is not None
            pf_b = flow_anchor[:, :, :3]
            pw_b = torch.sigmoid(flow_anchor[:, :, 3])
            T = dualflow2pose(
                xyz_src=action_bn3, xyz_tgt=anchor_bn3,
                flow_src=flow_a, flow_tgt=pf_b,
                weights_src=pw_a, weights_tgt=pw_b,
                return_transform3d=True,
                normalization_scehme='l1', training=True)
        else:
            T = flow2pose(
                xyz=action_bn3, flow=flow_a, weights=pw_a,
                return_transform3d=True,
                normalization_scehme='l1')
        
        return T

    @torch.no_grad()
    def inference(self, *input, **kwargs):
        return self.forward(*input, compute_loss=None)


class TwoStageFlowTransformer(ResidualFlow_DiffEmbTransformer):
    """两阶段流预测 + GRU 迭代精调 (方案 E).

    阶段 1: 继承父类 coarse_step — 标准粗流预测
    阶段 2: GRU 迭代修正 — 每步 SVD → 刚性流 → 误差 → GRU → delta_flow

    架构:
      Encoder (一次) ─→ coarse_step ─→ flow_0
         │                                  │
         └── action_embedding_tf ──────────[GRU]←── SVD(T) − predicted
                                            │
                                         delta_head → flow_{t+1} = flow_t + delta
    """

    def __init__(
        self,
        encoder_cfg,
        head_cfg,
        *args,
        num_refine_steps: int = 3,
        refine_hidden_dim: int = 128,
        fine_tune_trans: bool = False,
        **kwargs,
    ):
        super().__init__(encoder_cfg, head_cfg, *args, **kwargs)
        self.num_refine_steps = num_refine_steps
        self.refine_hidden_dim = refine_hidden_dim
        emb_dims = encoder_cfg.emb_dims

        # ── 上下文编码: [embedding(C) + pr(3) + flow(3)] → hidden ──
        context_in = emb_dims + 7
        # gn_groups = max(1, refine_hidden_dim // 16)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_in, refine_hidden_dim),
            # nn.GroupNorm(gn_groups, refine_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(refine_hidden_dim, refine_hidden_dim),
        )

        # ── GRU 时序记忆 ──
        self.gru_action = nn.GRUCell(refine_hidden_dim, refine_hidden_dim)
        self.gru_anchor = nn.GRUCell(refine_hidden_dim, refine_hidden_dim)

        # ── delta 预测头: hidden → delta_flow(3) + [delta_weight(1)] ──
        dim_flow = 4 if self.head_action.pred_weight else 3
        self.delta_head = nn.Sequential(
            nn.Linear(refine_hidden_dim, refine_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(refine_hidden_dim // 2, dim_flow),
        )
        self.fine_tune_trans = fine_tune_trans
        if fine_tune_trans:
            fine_tune_trans_context_in = emb_dims + 4 + 3
            # refine_hidden_dim = refine_hidden_dim * 2
            self.refine_trans_hidden_dim = refine_hidden_dim

            self.fine_tune_trans_context_encoder = nn.Sequential(
                nn.Linear(fine_tune_trans_context_in, refine_hidden_dim),
                # nn.GroupNorm(gn_groups, refine_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(refine_hidden_dim, refine_hidden_dim),
            )

            # ── GRU 时序记忆 ──
            self.fine_tune_trans_gru_action = nn.GRUCell(refine_hidden_dim, refine_hidden_dim)
            self.fine_tune_trans_gru_anchor = nn.GRUCell(refine_hidden_dim, refine_hidden_dim)

            # ── delta 预测头: hidden → delta_flow(3) + [delta_weight(1)] ──
            # dim_trans = 6
            # self.delta_rot_scale = 1.0 # 5 / 180 * torch.pi   # 可学习
            # self.delta_trans_scale = 1.0 #0.01
            self.fine_tune_trans_delta_head = nn.Sequential(
                nn.Linear(refine_hidden_dim, refine_hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(refine_hidden_dim // 2, dim_flow),
            )

    def get_parameters(self, module: str):
        """返回指定模块的参数列表，用于分层学习率。
        继承父类 get_parameters，并增加 "gru" 模块。
        """
        if module == "gru":
            params = list(self.context_encoder.parameters()) + \
                     list(self.gru_action.parameters()) + \
                     list(self.gru_anchor.parameters()) + \
                     list(self.delta_head.parameters())
            if self.fine_tune_trans:
                params += list(self.fine_tune_trans_context_encoder.parameters())
                params += list(self.fine_tune_trans_gru_action.parameters())
                params += list(self.fine_tune_trans_gru_anchor.parameters())
                params += list(self.fine_tune_trans_delta_head.parameters())
            return params
        return super().get_parameters(module)

    def _gru_step(self, gru_net, emb_tf, p_flow, error, hiddn):
        """GRU 迭代精调步骤
        Args:
            gru_net: GRUCell 网络
            emb_tf: Transformer embedding (B,N,C)
            p_flow: 预测的流 (B,N,4)
            error: 刚性流误差 (B,N,3)
            hiddn: GRU 隐状态 (B,N,H)
        Returns:
            delta_flow: GRU 输出的流增量 (B,N,4)
            hiddn: 更新后的 GRU 隐状态 (B,N,H)
        """
        B, N, _ = p_flow.shape
        assert emb_tf.size(1) == hiddn.size(1) == N, \
            f"Mismatch in number of points: emb_tf {emb_tf.size(1)}, hiddn {hiddn.size(1)}"
        ctx_a = torch.cat(
            [
                emb_tf,
                p_flow,
                error
            ], dim=-1)     # (B,N,C+3+4)
        ctx_a = self.context_encoder(ctx_a)  # (B,N,H)

        # GRU 更新
        ctx_flat = ctx_a.reshape(B * N, -1)
        hid_flat = hiddn.reshape(B * N, -1)
        h_new = gru_net(ctx_flat, hid_flat)
        hiddn = h_new.reshape(B, N, -1)
        delta_a = self.delta_head(hiddn)  # (B,N,4)
        return delta_a, hiddn

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, *input, **kwargs):
        # ═══════════════ 阶段 1: 粗预测 (复用父类) ══════════════
        outputs = super().forward(*input, **kwargs)

        if self.num_refine_steps <= 0:
            return outputs

        shared_args = outputs['shared_args']
        action_embedding = shared_args[2]
        # Pop TF embedding, which will not be used at next stage
        act_emb_tf = shared_args.pop(4).detach().transpose(1, 2).contiguous()
        anch_emb_tf = shared_args.pop(4).detach().transpose(1, 2).contiguous()

        has_anchor = ("flow_anchor" in outputs
                      and outputs["flow_anchor"] is not None)
        down_sample = shared_args[0].shape[2] != action_embedding.shape[2]

        action_bn3 = shared_args[0] if not down_sample else shared_args[-4]
        action_bn3 = action_bn3.permute(0, 2, 1).contiguous()
        anchor_bn3 = shared_args[1].permute(0, 2, 1).contiguous() if has_anchor else None
        if has_anchor and down_sample:
            anchor_bn3 = shared_args[-3].permute(0, 2, 1).contiguous()

        if "compute_loss" in kwargs:
            coarse_loss, _ = kwargs["compute_loss"](outputs)

        hook_trans_T = []
        observing_deltaT = []
        # ═══════════════ 阶段 2: GRU 迭代精调 ═══════════════
        flow_action = outputs["flow_action"]                 # (B, N, 4)
        flow_anchor = outputs["flow_anchor"] if has_anchor else None

        B, N, _ = flow_action.shape
        # 初始化 GRU 隐状态
        h_a = torch.zeros(B, N, self.refine_hidden_dim,
                          device=flow_action.device)
        h_b = torch.zeros(B, N, self.refine_hidden_dim,
                          device=flow_action.device) if has_anchor else None
        if self.fine_tune_trans:
            h_trans_a = torch.zeros_like(h_a)
            h_trans_b = torch.zeros_like(h_b) if has_anchor else None

        refined_loss_list = []
        for step in range(self.num_refine_steps):
            # ── SVD: 从当前 flow 解位姿 (detach, 不穿梯度) ──
            fa = flow_action.detach()
            pf_a = fa[:, :, :3]                                  # (B, N, 3)
            T = self._solve_transformation(
                action_bn3, fa,
                has_anchor=has_anchor,
                anchor_bn3=anchor_bn3,
                flow_anchor=flow_anchor,
            )
            hook_trans_T.append(T)
            # ── Action 侧精调 ──
            coarsed_act = T.transform_points(action_bn3)
            rigid_a = coarsed_act - action_bn3  # (B,N,3)
            error_a = rigid_a - pf_a            # (B,N,3)
            delta_a, h_a = self._gru_step(self.gru_action, act_emb_tf, fa, error_a, h_a)
            flow_action = delta_a + flow_action

            # ── Anchor 侧精调 (对称) ──
            if has_anchor:
                assert flow_anchor is not None
                fb = flow_anchor.detach()
                coarsed_anchor = T.inverse().transform_points(anchor_bn3)
                rigid_b = coarsed_anchor - anchor_bn3
                error_b = rigid_b - fb[:, :, :3]
                delta_b, h_b = self._gru_step(self.gru_anchor, anch_emb_tf, fb, error_b, h_b)
                flow_anchor = flow_anchor + delta_b

            if self.fine_tune_trans:
                tgt_pt = kwargs.get("gt_act_target", anchor_bn3)
                tgt_pt = knn_points_with_normals(
                    coarsed_act, tgt_pt,
                    n1=estimate_pointcloud_normals(coarsed_act, 20),
                    n2=estimate_pointcloud_normals(tgt_pt, 20),
                    K=1, return_nn=True)[-1]  # (B,N,k,3)
                if tgt_pt.dim() == 4:
                    tgt_pt = tgt_pt.squeeze(2)  # (B,N,3)
                # key_point_error: torch.Tensor = chamfer_distance(
                #     action_bn3, tgt_pt,
                #     batch_reduction=None, point_reduction=None,
                #     single_directional=True)[0]
                key_point_error = tgt_pt - coarsed_act
                ctx_act = torch.cat(
                    [
                        act_emb_tf,
                        flow_action,
                        # anchor_embedding.detach(),
                        key_point_error,
                    ], dim=-1)
                ctx_act = self.fine_tune_trans_context_encoder(ctx_act)  # (B, N, H)
                flat = ctx_act.reshape(B * N, -1)
                h_flat = h_trans_a.reshape(B * N, -1)
                new_ = self.gru_action(flat, h_flat)
                h_trans_a = new_.reshape(B, N, -1)

                delta_a = self.fine_tune_trans_delta_head(h_trans_a)              # (B,N,4)
                flow_action = flow_action + delta_a
                
                if has_anchor:
                    tgt_pt = kwargs.get("gt_anch_target", action_bn3)
                    tgt_pt = knn_points_with_normals(
                        coarsed_anchor, tgt_pt,
                        n1=estimate_pointcloud_normals(coarsed_anchor, 20),
                        n2=estimate_pointcloud_normals(tgt_pt, 20),
                        K=1, return_nn=True)[-1]  # (B,N,k,3)
                    if tgt_pt.dim() == 4:
                        tgt_pt = tgt_pt.squeeze(2)  # (B,N,3)

                    # key_point_error_b: torch.Tensor = chamfer_distance(
                    #     anchor_bn3, tgt_pt,
                    #     batch_reduction=None, point_reduction=None,
                    #     single_directional=True)[0]
                    key_point_error_b = tgt_pt - coarsed_anchor
                    ctx_anch = torch.cat(
                        [
                            anch_emb_tf,
                            flow_anchor,
                            key_point_error_b,
                        ], dim=-1)
                    ctx_anch = self.fine_tune_trans_context_encoder(ctx_anch)  # (B, N, H)
                    flat = ctx_anch.reshape(B * N, -1)
                    h_flat = h_trans_b.reshape(B * N, -1)
                    new_ = self.gru_anchor(flat, h_flat)
                    h_trans_b = new_.reshape(B, N, -1)
                    delta_b = self.fine_tune_trans_delta_head(h_trans_b)
                    flow_anchor = flow_anchor + delta_b

            outputs["flow_action"] = flow_action
            if has_anchor:
                outputs["flow_anchor"] = flow_anchor
            if "compute_loss" in kwargs:
                refined_loss, _ = kwargs["compute_loss"](outputs)
                refined_loss_list.append(sum(refined_loss) / self.num_refine_steps)

        # Hook to save trans, which are used to observe the T each fine-tune step
        outputs.update(hook_trans_T=hook_trans_T)
        if len(observing_deltaT) > 0:
            outputs.update(observing_deltaT=observing_deltaT)

        if "compute_loss" in kwargs:
            outputs.update(coarse_loss=coarse_loss)
            outputs.update(refined_loss=refined_loss_list)

        shared_args[-2] = shared_args[-2] + flow_action[:, :, :3].permute(0, 2, 1).contiguous()
        if has_anchor:  # otherwise, shared_args[-1] is None
            shared_args[-1] = shared_args[-1] + flow_anchor[:, :, :3].permute(0, 2, 1).contiguous()
        return outputs

    @torch.no_grad()
    def inference(self, *input, **kwargs):
        return self.forward(*input, **kwargs)


class CascadeFlowTransformer(nn.Module):
    def __init__(
        self,
        encoder_cfg,
        head_cfg, 
        stage_num: int = 2,
        num_refine_steps: int = 0,
        **kwargs
    ):
        super().__init__()
        self.emb_nn_action = create_embedding_network(encoder_cfg)
        self.emb_nn_anchor = create_embedding_network(encoder_cfg)
        self.stage_num = stage_num
        block_type = TwoStageFlowTransformer if num_refine_steps > 0 \
            else ResidualFlow_DiffEmbTransformer
        self.blocks = nn.ModuleList()
        for i in range(stage_num):
            self.blocks.append(
                block_type(
                    encoder_cfg,
                    head_cfg,
                    stage=i,
                    is_final=(i == stage_num - 1),
                    **kwargs
                )
            )
            self.blocks[-1].emb_nn_action = self.emb_nn_action
            self.blocks[-1].emb_nn_anchor = self.emb_nn_anchor

    def get_parameters(self, module: str):
        """返回指定模块的参数列表，用于分层学习率。
        Cascade 中 emb 模块为所有 stage 共享，其余模块逐 stage 收集。
        """
        if module == "emb":
            # 共享的 embedding 网络（仅一份）
            params = list(self.emb_nn_action.parameters()) + \
                     list(self.emb_nn_anchor.parameters())
            # 各 block 自己的 pos_encoder / feature_channel_encoder
            for block in self.blocks:
                block_params = block.get_parameters("emb")
                # 排除已被 emb_nn_action/anchor 覆盖的共享参数
                shared_ids = {id(p) for p in params}
                for p in block_params:
                    if id(p) not in shared_ids:
                        params.append(p)
                        shared_ids.add(id(p))
            return params
        elif module in ("backbone", "head", "gru"):
            params = []
            seen_ids = set()
            for block in self.blocks:
                for p in block.get_parameters(module):
                    if id(p) not in seen_ids:
                        params.append(p)
                        seen_ids.add(id(p))
            return params
        else:
            return self.blocks[0].get_parameters(module)  # 抛出 ValueError

    def forward(self, *input, **kwargs):
        staged_coarse_loss = []
        staged_refined_loss = []
        staged_hook_trans_T = []
        outputs = {}
        assert "compute_loss" in kwargs
        for i, block in enumerate(self.blocks):
            outputs = block(*input, **kwargs)
            input = outputs.pop("shared_args")
            if len(input) == 12:
                input.pop(4)
                input.pop(4)
            if "refined_loss" not in outputs:
                outputs.update(
                    refined_loss=[sum(kwargs["compute_loss"](outputs)[0])]
                )
            staged_coarse_loss += outputs.pop(
                "coarse_loss", [torch.zeros(1,).to(input[0].device)])
            staged_refined_loss += outputs.pop(
                "refined_loss", [torch.zeros(1,).to(input[0].device)])
            current_hook_trans_T = outputs.pop("hook_trans_T", None)
            if current_hook_trans_T is None:
                current_hook_trans_T = [self._solve_transformation(
                    input, outputs
                )]
            staged_hook_trans_T += current_hook_trans_T

        outputs.update(
            coarse_loss=staged_coarse_loss,
            refined_loss=staged_refined_loss,
            hook_trans_T=staged_hook_trans_T
        )
        return outputs

    def _solve_transformation(self, shared_args, outputs):
        action_embedding = shared_args[2]

        has_anchor = ("flow_anchor" in outputs
                      and outputs["flow_anchor"] is not None)
        down_sample = shared_args[0].shape[2] != action_embedding.shape[2]

        action_bn3 = shared_args[0] if not down_sample else shared_args[-4]
        action_bn3 = action_bn3.permute(0, 2, 1).contiguous()
        anchor_bn3 = shared_args[1].permute(0, 2, 1).contiguous() if has_anchor else None
        if has_anchor and down_sample:
            anchor_bn3 = shared_args[-3].permute(0, 2, 1).contiguous()
        
        T = ResidualFlow_DiffEmbTransformer._solve_transformation(
            action_bn3, outputs["flow_action"],
            has_anchor=has_anchor,
            anchor_bn3=anchor_bn3,
            flow_anchor=outputs["flow_anchor"] if has_anchor else None,
        )
        return T

    @torch.no_grad()
    def inference(self, *input, **kwargs):
        for i, block in enumerate(self.blocks):
            outputs = block(*input)
            input = outputs.pop("shared_args")
            if len(input) == 12:
                input.pop(4)
                input.pop(4)
        
        return outputs


class ModelConfig(Protocol):
    model_type: str


@dataclass
class ResidualFlowDiffEmbTransformerConfig:
    model_type: ClassVar[str] = "residual_flow_diff_emb_transformer"

    encoder: Any
    head: Any

    cycle: bool
    center_feature: bool
    freeze_embnn: bool
    return_attn: bool

    # Multilateration
    multilaterate: bool
    mlat_sample: bool
    mlat_nkps: bool

    # Extra channels.
    feature_channels: int
    conditional: bool

    dropout: float = 0.0
    pos_encoding: bool = False
    n_blocks: int = 1
    attn_mode: str = "torch_attn"
    # Fine-tune
    num_refine_steps: int = 0
    refine_hidden_dim: int = 128
    fine_tune: bool = False
    # Cascade
    stage_num: int = 1


@dataclass
class CorrespondenceFlowDiffEmbMLPConfig:
    model_type: ClassVar[str] = "correspondence_flow_diff_emb_mlp"

    encoder: Any

    cycle: bool
    center_feature: bool


def create_network(cfg: ModelConfig) -> nn.Module:
    # Create the network
    if cfg.model_type == "residual_flow_diff_emb_transformer":
        r_cfg = cast(ResidualFlowDiffEmbTransformerConfig, cfg)
        network: nn.Module = ResidualFlow_DiffEmbTransformer(
            encoder_cfg=r_cfg.encoder,
            head_cfg=r_cfg.head,
            cycle=r_cfg.cycle,
            center_feature=r_cfg.center_feature,
            freeze_embnn=r_cfg.freeze_embnn,
            return_attn=r_cfg.return_attn,
            multilaterate=r_cfg.multilaterate,
            mlat_sample=r_cfg.mlat_sample,
            mlat_nkps=r_cfg.mlat_nkps,
            feature_channels=r_cfg.feature_channels,
            conditional=r_cfg.conditional,
            dropout=r_cfg.dropout,
            pos_encoding=r_cfg.pos_encoding,
            n_blocks=r_cfg.n_blocks,
            attn_mode=r_cfg.attn_mode,
            fine_tune=getattr(cfg, 'fine_tune', False),
            weight_beta=getattr(cfg, 'weight_beta', 0.5),
        )
    elif cfg.model_type == "two_stage_flow_transformer":
        r_cfg = cast(ResidualFlowDiffEmbTransformerConfig, cfg)
        network: nn.Module = TwoStageFlowTransformer(
            encoder_cfg=r_cfg.encoder,
            head_cfg=r_cfg.head,
            cycle=r_cfg.cycle,
            center_feature=r_cfg.center_feature,
            freeze_embnn=r_cfg.freeze_embnn,
            return_attn=r_cfg.return_attn,
            multilaterate=r_cfg.multilaterate,
            mlat_sample=r_cfg.mlat_sample,
            mlat_nkps=r_cfg.mlat_nkps,
            feature_channels=r_cfg.feature_channels,
            conditional=r_cfg.conditional,
            dropout=r_cfg.dropout,
            pos_encoding=r_cfg.pos_encoding,
            n_blocks=r_cfg.n_blocks,
            attn_mode=r_cfg.attn_mode,
            num_refine_steps=r_cfg.num_refine_steps,
            refine_hidden_dim=r_cfg.refine_hidden_dim,
            fine_tune_trans=r_cfg.fine_tune,
        )
    elif cfg.model_type == "correspondence_flow_diff_emb_mlp":
        c_cfg = cast(CorrespondenceFlowDiffEmbMLPConfig, cfg)
        network = CorrespondenceFlow_DiffEmbMLP(
            encoder_cfg=c_cfg.encoder,
            cycle=c_cfg.cycle,
            center_feature=c_cfg.center_feature,
        )
    elif cfg.model_type == "cascade_flow_transformer":
        r_cfg = cast(ResidualFlowDiffEmbTransformerConfig, cfg)
        network = CascadeFlowTransformer(
            encoder_cfg=r_cfg.encoder,
            stage_num=r_cfg.stage_num,
            num_refine_steps=r_cfg.num_refine_steps,
            head_cfg=r_cfg.head,
            cycle=r_cfg.cycle,
            center_feature=r_cfg.center_feature,
            freeze_embnn=r_cfg.freeze_embnn,
            return_attn=r_cfg.return_attn,
            multilaterate=r_cfg.multilaterate,
            mlat_sample=r_cfg.mlat_sample,
            mlat_nkps=r_cfg.mlat_nkps,
            feature_channels=r_cfg.feature_channels,
            conditional=r_cfg.conditional,
            dropout=r_cfg.dropout,
            pos_encoding=r_cfg.pos_encoding,
            n_blocks=r_cfg.n_blocks,
            attn_mode=r_cfg.attn_mode,
            refine_hidden_dim=r_cfg.refine_hidden_dim,
            fine_tune_trans=r_cfg.fine_tune,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")

    return network
