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
from taxpose.nets.raw_dgcnn import DGCNN4TaxPose, DGCNN_VAE
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
        attn_mode="torch.nn",
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
        self.emb_nn_anchor = create_embedding_network(encoder_cfg)
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

    def _action_embedding(self, points, allow_down=False):
        """
        Embedding function for the action point cloud.
        Args:
            points: B,3,num_points
            allow_down: Allow downsampling.
        """
        self.emb_nn_action.eval()
        points = points - points.mean(dim=2, keepdim=True)
        embedding = self.emb_nn_action(points, down=allow_down)
        # embedding = F.normalize(embedding, dim=1)
        return embedding

    def _anchor_embedding(self, points, allow_down=False):
        self.emb_nn_anchor.eval()
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

    def _backbone(self, action_embedding, anchor_embedding, action_pt_pos, anchor_pt_pos):
        action_embedding = action_embedding.transpose(2, 1).contiguous()
        anchor_embedding = anchor_embedding.transpose(2, 1).contiguous()

        transformer_action_outputs = self.transformer_action(
            action_embedding, anchor_embedding
        )
        transformer_anchor_outputs = self.transformer_anchor(
            anchor_embedding, action_embedding
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

    def forward(self, *input, compute_loss=None):
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
        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn
        ) = self._backbone(action_embedding, anchor_embedding, action_pt_pos, anchor_pt_pos)

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
        #  *************************DEBUG*************************
        outputs.update(attns=action_attn)
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

    def set_residual_on(self, on):
        self.head_action.residual_on = on
        self.head_anchor.residual_on = on

    @torch.no_grad()
    def inference(self, *input, **kwargs):
        return self.forward(*input, compute_loss=None)


class Flow_DiffEmbTransformer(ResidualFlow_DiffEmbTransformer):
    def forward(self, *input):
        action_points = input[0].permute(0, 2, 1)[:, :3]  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)[:, :3]

        action_points_dmean = action_points - action_points.mean(dim=2, keepdim=True)
        anchor_points_dmean = anchor_points - anchor_points.mean(dim=2, keepdim=True)

        # mean center point cloud before DGCNN
        if not self.center_feature:
            action_points_dmean = action_points
            anchor_points_dmean = anchor_points
        if self.freeze_embnn:
            self.emb_nn_action.eval()
        act_down_sample, anch_down_sample = None, None
        with torch.set_grad_enabled(not self.freeze_embnn):
            action_embedding = self.emb_nn_action(action_points_dmean)
            if isinstance(action_embedding, tuple):
                action_embedding, pts = action_embedding
                act_down_sample = pts + action_points.mean(dim=2, keepdim=True)
                act_down_sample = act_down_sample
                # action_points = pts + action_points.mean(dim=2, keepdim=True)
            anchor_embedding = self.emb_nn_anchor(anchor_points_dmean)
            if isinstance(anchor_embedding, tuple):
                anchor_embedding, pts = anchor_embedding
                anch_down_sample = pts + anchor_points.mean(dim=2, keepdim=True)
                anch_down_sample = anch_down_sample
                # anchor_points = pts + anchor_points.mean(dim=2, keepdim=True)
            action_embedding = F.normalize(action_embedding, dim=1)
            anchor_embedding = F.normalize(anchor_embedding, dim=1)
        # if self.freeze_embnn:
        #     action_embedding = action_embedding.detach()
        #     anchor_embedding = anchor_embedding.detach()

        if self.feature_channels > 0:
            # Add a symmetry label to the embeddings.
            action_features = input[2].permute(0, 2, 1)
            anchor_features = input[3].permute(0, 2, 1)

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
        if self.pos_encoding:
            action_pt_pos = self.pos_encoder(action_points)  # B,C,N
            anchor_pt_pos = self.pos_encoder(anchor_points)
            action_embedding += F.normalize(action_pt_pos)
            anchor_embedding += F.normalize(anchor_pt_pos)
        
        transformer_action_outputs = self.transformer_action(
            action_embedding, anchor_embedding
        )
        transformer_anchor_outputs = self.transformer_anchor(
            anchor_embedding, action_embedding
        )
        action_embedding_tf = transformer_action_outputs["src_embedding"]
        action_attn = transformer_action_outputs["src_attn"]
        anchor_embedding_tf = transformer_anchor_outputs["src_embedding"]
        anchor_attn = transformer_anchor_outputs["src_attn"]

        if not self.return_attn:
            action_attn = None
            anchor_attn = None
        # 理论上， action_embedding_tf = action_embedding + residual(action_embedding)
        # action_embedding_tf = action_embedding + F.normalize(action_embedding_tf, dim=1)
        # anchor_embedding_tf = anchor_embedding + F.normalize(anchor_embedding_tf, dim=1)

        if action_attn is not None:
            action_attn = action_attn.mean(dim=1)  # b, h，N, M -> b, N, M

        if self.pos_encoding:
            action_embedding_tf += action_pt_pos
            anchor_embedding_tf += anchor_pt_pos

        del transformer_action_outputs, transformer_anchor_outputs
        
        head_action_output = self.head_action(
            action_embedding_tf,
            anchor_embedding,
            action_points,
            anchor_points,
            act_down_sample,
            anch_down_sample,
            scores=action_attn
        )
        flow_action = head_action_output["full_flow"].permute(0, 2, 1)
        residual_flow_action = head_action_output["residual_flow"].permute(0, 2, 1)
        corr_flow_action = head_action_output["corr_flow"].permute(0, 2, 1)
        corr_points_action = head_action_output["corr_points"].permute(0, 2, 1)

        outputs = {
            "flow_action": flow_action,
            "residual_flow_action": residual_flow_action,
            "corr_flow_action": corr_flow_action,
            "corr_points_action": corr_points_action,
            "act_down_sample": (
                None if act_down_sample is None else act_down_sample.permute(0, 2, 1)
            ),
        }

        if "P_A" in head_action_output:
            original_points_action = head_action_output["P_A"].permute(0, 2, 1)
            outputs["original_points_action"] = original_points_action
            outputs["sampled_ixs_action"] = head_action_output["A_ixs"]

        del head_action_output

        if self.cycle:
            anchor_attn = anchor_attn.mean(dim=1)
            head_anchor_output = self.head_anchor(
                anchor_embedding_tf,
                action_embedding,
                anchor_points,
                action_points,
                anch_down_sample,
                act_down_sample,
                scores=anchor_attn,
            )
            flow_anchor = head_anchor_output["full_flow"].permute(0, 2, 1)
            residual_flow_anchor = head_anchor_output["residual_flow"].permute(0, 2, 1)
            corr_flow_anchor = head_anchor_output["corr_flow"].permute(0, 2, 1)
            corr_points_anchor = head_anchor_output["corr_points"].permute(0, 2, 1)

            outputs = {
                **outputs,
                "flow_anchor": flow_anchor,
                "residual_flow_anchor": residual_flow_anchor,
                "corr_flow_anchor": corr_flow_anchor,
                "corr_points_anchor": corr_points_anchor,
                "anch_down_sample": (
                    None
                    if anch_down_sample is None
                    else anch_down_sample.permute(0, 2, 1)
                ),
            }

            if "P_A" in head_anchor_output:
                original_points_anchor = head_anchor_output["P_A"].permute(0, 2, 1)
                outputs["original_points_anchor"] = original_points_anchor
                outputs["sampled_ixs_anchor"] = head_anchor_output["A_ixs"]

            del head_anchor_output
        #  *************************DEBUG*************************
        outputs.update(attns=action_attn)
        return outputs
       

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
        context_in = emb_dims + 6
        # gn_groups = max(1, refine_hidden_dim // 16)
        self.context_encoder = nn.Sequential(
            nn.Conv1d(context_in, refine_hidden_dim, 1),
            # nn.GroupNorm(gn_groups, refine_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(refine_hidden_dim, refine_hidden_dim, 1),
        )

        # ── GRU 时序记忆 ──
        self.gru_action = nn.GRUCell(refine_hidden_dim, refine_hidden_dim)
        self.gru_anchor = nn.GRUCell(refine_hidden_dim, refine_hidden_dim)

        # ── delta 预测头: hidden → delta_flow(3) + [delta_weight(1)] ──
        dim_flow = 4 if self.head_action.pred_weight else 3
        self.delta_head = nn.Sequential(
            nn.Conv1d(refine_hidden_dim, refine_hidden_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(refine_hidden_dim // 2, dim_flow, 1),
        )
        self.fine_tune_trans = fine_tune_trans
        if fine_tune_trans:
            fine_tune_trans_context_in = emb_dims * 2 + 4
            # refine_hidden_dim = refine_hidden_dim * 2
            self.refine_trans_hidden_dim = refine_hidden_dim

            self.fine_tune_trans_context_encoder = nn.Sequential(
                nn.Conv1d(fine_tune_trans_context_in, 2 * refine_hidden_dim, 1),
                # nn.GroupNorm(gn_groups, refine_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(2 * refine_hidden_dim, refine_hidden_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(refine_hidden_dim, refine_hidden_dim, 1),
                nn.AdaptiveMaxPool1d(1)  # (B, H, N) → (B, H, 1)
            )

            # ── GRU 时序记忆 ──
            self.fine_tune_trans_gru_action = nn.GRUCell(refine_hidden_dim, refine_hidden_dim)
            self.fine_tune_trans_gru_anchor = nn.GRUCell(refine_hidden_dim, refine_hidden_dim)

            # ── delta 预测头: hidden → delta_flow(3) + [delta_weight(1)] ──
            dim_trans = 6
            self.delta_rot_scale = 5 / 180 * torch.pi   # 可学习
            self.delta_trans_scale = 0.01
            self.fine_tune_trans_delta_head = nn.Sequential(
                nn.Linear(refine_hidden_dim * 2, refine_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(refine_hidden_dim, refine_hidden_dim // 2, 1),
                nn.ReLU(inplace=True),
                nn.Linear(refine_hidden_dim // 2, dim_trans, 1),
                nn.Tanh()
            )

    def forward(self, *input, **kwargs):
        # ═══════════════ 阶段 1: 粗预测 (复用父类) ══════════════
        outputs = super().forward(*input, **kwargs)

        if self.num_refine_steps <= 0:
            return outputs

        shared_args = outputs['shared_args']
        action_embedding = shared_args[2]
        anchor_embedding = shared_args[3]
        # Pop TF embedding, which will not be used at next stage
        act_emb_tf = shared_args.pop(4)
        anchor_emb_tf = shared_args.pop(4)

        has_anchor = ("flow_anchor" in outputs
                      and outputs["flow_anchor"] is not None)
        down_sample = input[0].shape[1] != action_embedding.shape[2]

        action_bn3 = input[0] if not down_sample \
            else shared_args[-4].permute(0, 2, 1).contiguous()
        anchor_bn3 = input[1] if has_anchor else None
        if has_anchor and down_sample:
            anchor_bn3 = shared_args[-3].permute(0, 2, 1).contiguous()
        if "compute_loss" in kwargs:
            coarse_loss, _ = kwargs["compute_loss"](outputs)

        hook_trans_T = []
        # ═══════════════ 阶段 2: GRU 迭代精调 ═══════════════
        flow_action = outputs["flow_action"]                 # (B, N, 4)
        flow_anchor = outputs["flow_anchor"] if has_anchor else None

        B, N, _ = flow_action.shape
        # 初始化 GRU 隐状态
        h_a = torch.zeros(B, self.refine_hidden_dim, N,
                          device=flow_action.device)
        h_b = torch.zeros(B, self.refine_hidden_dim, N,
                          device=flow_action.device) if has_anchor else None
        if self.fine_tune_trans:
            h_trans_a = torch.zeros(B, self.refine_trans_hidden_dim,
                            device=flow_action.device)
            h_trans_b = torch.zeros(B, self.refine_trans_hidden_dim,
                            device=flow_action.device) if has_anchor else None

        refined_loss_list = []
        reg_loss = 0.0
        for step in range(self.num_refine_steps):
            # ── SVD: 从当前 flow 解位姿 (detach, 不穿梯度) ──
            fa = flow_action.detach()
            pf_a = fa[:, :, :3]                                  # (B, N, 3)
            pw_a = torch.sigmoid(fa[:, :, 3])                    # (B, N)

            if has_anchor:
                fb = flow_anchor.detach()
                pf_b = fb[:, :, :3]
                pw_b = torch.sigmoid(fb[:, :, 3])
                T = dualflow2pose(
                    xyz_src=action_bn3, xyz_tgt=anchor_bn3,
                    flow_src=pf_a, flow_tgt=pf_b,
                    weights_src=pw_a, weights_tgt=pw_b,
                    return_transform3d=True,
                    normalization_scehme='l1', training=True)
            else:
                T = flow2pose(
                    xyz=action_bn3, flow=pf_a, weights=pw_a,
                    return_transform3d=True,
                    normalization_scehme='l1')
            # ── Action 侧精调 ──
            rigid_a = T.transform_points(action_bn3) - action_bn3  # (B,N,3)
            error_a = rigid_a - pf_a                                      # (B,N,3)
            ctx_a = torch.cat(
                [
                    act_emb_tf.detach(),
                    pf_a.transpose(1, 2),
                    error_a.transpose(1, 2)
                ], dim=1)     # (B,C+6,N)
            ctx_a = self.context_encoder(ctx_a)                           # (B,H,N)

            # GRU 更新
            ctx_flat = ctx_a.permute(0, 2, 1).reshape(B * N, -1)
            hid_flat = h_a.permute(0, 2, 1).reshape(B * N, -1)
            h_new = self.gru_action(ctx_flat, hid_flat)
            h_a = h_new.reshape(B, N, -1).permute(0, 2, 1)

            delta_a = self.delta_head(h_a).permute(0, 2, 1)              # (B,N,4)
            flow_action = flow_action + delta_a

            # ── Anchor 侧精调 (对称) ──
            if has_anchor:
                rigid_b = T.inverse().transform_points(
                    anchor_bn3) - anchor_bn3
                error_b = rigid_b - pf_b
                ctx_b = torch.cat(
                    [
                        anchor_emb_tf.detach(),
                        pf_b.transpose(1, 2),
                        error_b.transpose(1, 2),
                    ], dim=1)
                ctx_b = self.context_encoder(ctx_b)

                ctx_flat_b = ctx_b.permute(0, 2, 1).reshape(B * N, -1)
                hid_flat_b = h_b.permute(0, 2, 1).reshape(B * N, -1)
                h_new_b = self.gru_anchor(ctx_flat_b, hid_flat_b)
                h_b = h_new_b.reshape(B, N, -1).permute(0, 2, 1)

                delta_b = self.delta_head(h_b).permute(0, 2, 1)
                flow_anchor = flow_anchor + delta_b
            
            outputs["flow_action"] = flow_action
            if has_anchor:
                outputs["flow_anchor"] = flow_anchor
            if "compute_loss" in kwargs:
                refined_loss, _ = kwargs["compute_loss"](outputs)
                refined_loss_list.append(sum(refined_loss) / self.num_refine_steps)

            if self.fine_tune_trans:
                ctx_act = torch.cat(
                    [
                        act_emb_tf.detach(),
                        anchor_embedding.detach(),
                        flow_action.detach().transpose(1, 2),
                    ], dim=1)
                ctx_act = self.fine_tune_trans_context_encoder(ctx_act).squeeze(-1)          # (B, H)
                h_trans_a = self.fine_tune_trans_gru_action(ctx_act, h_trans_a)  # (B, H)
                
                if has_anchor:
                    ctx_anch = torch.cat(
                        [
                            anchor_emb_tf.detach(),
                            action_embedding.detach(),
                            flow_anchor.detach().transpose(1, 2),
                        ], dim=1)
                    ctx_anch = self.fine_tune_trans_context_encoder(ctx_anch).squeeze(-1)          # (B, H)
                    h_trans_b = self.fine_tune_trans_gru_anchor(ctx_anch, h_trans_b)
                
                delta_se3 = self.fine_tune_trans_delta_head(
                    torch.cat([h_trans_a, h_trans_b], dim=1))         # (B, 6)
                # NOTE: 使用 6D元素表达旋转会导致初始角度太大
                # delta_R = axis_angle_to_matrix(delta_se3[:, :3] * self.delta_rot_scale)  # (B, 3, 3)
                # Try ℝ³ → S³ (轴角 → 四元数)
                epsilon = delta_se3[:, :3] * self.delta_rot_scale
                theta = epsilon.norm(p=2, dim=-1, keepdim=True)     # (B, 1)
                safe_theta = torch.where(theta < 1e-8, torch.ones_like(theta), theta)
                axis = epsilon / safe_theta                          # (B, 3)
                cos_half = torch.cos(theta / 2.0)
                sin_half = torch.sin(theta / 2.0)
                delta_q = torch.cat([cos_half, sin_half * axis], dim=-1)  # (B, 4)
                delta_R = quaternion_to_matrix(delta_q)
                delta_t = delta_se3[:, 3:] * self.delta_trans_scale       # (B, 3)
                # reg_loss = 0.01 * (
                #     delta_se3[:, :3].norm(dim=-1).mean() +   # 旋转角度范数
                #     delta_se3[:, 3:].norm(dim=-1).mean()     # 平移范数
                # )
                delta_T = Rotate(delta_R).translate(delta_t)
                T = T.compose(delta_T)
                flow_act_trans = T.transform_normals(flow_action[:, :, :3])
                flow_action = torch.cat([flow_act_trans, flow_action[:, :, 3:]], dim=-1)
                if has_anchor:
                    flow_anch_trans = T.transform_normals(flow_anchor[:, :, :3])
                    flow_anchor = torch.cat([flow_anch_trans, flow_anchor[:, :, 3:]], dim=-1)
                outputs['tuned_T'] = T

                outputs["flow_action"] = flow_action
                if has_anchor:
                    outputs["flow_anchor"] = flow_anchor
                if "compute_loss" in kwargs and step < self.num_refine_steps - 1:
                    # 最后一次的精调损失由外部计算，保证统一接口
                    refined_loss, _ = kwargs["compute_loss"](outputs)
                    refined_loss_list.append(sum(refined_loss) + reg_loss)

            hook_trans_T.append(T)
        # Hook to save trans, which are used to observe the T each fine-tune step
        outputs.update(hook_trans_T=hook_trans_T)

        if "compute_loss" in kwargs:
            outputs.update(coarse_loss=coarse_loss)
            outputs.update(refined_loss=refined_loss_list)

        shared_args[-2] = shared_args[-2] + flow_action[:, :, :3].permute(0, 2, 1).contiguous()
        if has_anchor:  # otherwise, shared_args[-1] is None
            shared_args[-1] = shared_args[-1] + flow_anchor[:, :, :3].permute(0, 2, 1).contiguous()
        return outputs

    @torch.no_grad()
    def inference(self, *input, **kwargs):
        return self.forward(*input, compute_loss=None)


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

    def forward(self, *input, **kwargs):
        staged_coarse_loss = []
        staged_refined_loss = []
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
            staged_coarse_loss += outputs.pop("coarse_loss", [0.0])
            staged_refined_loss += outputs.pop("refined_loss", [0.0])

        outputs.update(
            coarse_loss=staged_coarse_loss,
            refined_loss=staged_refined_loss
        )
        return outputs

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
    elif cfg.model_type == "direct_correspondence_points_prediction":
        r_cfg = cast(ResidualFlowDiffEmbTransformerConfig, cfg)
        network: nn.Module = Flow_DiffEmbTransformer(
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
            pos_encoding=cfg.pos_encoding,
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
