#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Pulled from DCP
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F

from taxpose.nets.transformer_flow_pm import CustomTransformer
from taxpose.nets.raw_dgcnn import knn
from taxpose.nets.gemo_fea import ManualPointWiseGemoFea
from taxpose.utils.se3 import dualflow2pose, flow2pose
from taxpose.nets.tf_utils import key_point, knn_query_with_embedding
from taxpose.nets import create_head, HeadConfig
from taxpose.nets import create_embedding_network


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
        if freeze_embnn:
            self.emb_nn_anchor = create_embedding_network(encoder_cfg)
        else:
            self.emb_nn_anchor = self.emb_nn_action
        emb_dims = encoder_cfg.emb_dims
        self.freeze_embnn = freeze_embnn
        if freeze_embnn:
            self.emb_nn_action.requires_grad_(False)
            self.emb_nn_anchor.requires_grad_(False)
        self.center_feature = center_feature
        self.return_attn = return_attn
        self.conditional = conditional

        self.fine_tune = fine_tune
        self.weight_beta = weight_beta
        self.stage = stage
        self.is_final_stage = is_final
        self.knn_in_tf = "knn" in attn_mode
        self.knn_mode = kwargs.get("knn_mode", "emb")
        self.point_augmented = kwargs.get("point_augmented", False)
        self.point_ffn_mode = kwargs.get("point_ffn_mode", "none")

        self.transformer_action = CustomTransformer(
            emb_dims=emb_dims,
            n_blocks=n_blocks,
            dropout=dropout,
            ff_dims=4*emb_dims,
            n_heads=emb_dims//64,
            return_attn=self.return_attn,
            bidirectional=False,
            attn_mode=attn_mode,
            point_ffn_mode=self.point_ffn_mode,
            num_points=encoder_cfg.output_num,
        )
        self.transformer_anchor = CustomTransformer(
            emb_dims=emb_dims,
            n_blocks=n_blocks,
            dropout=dropout,
            ff_dims=4*emb_dims,
            n_heads=emb_dims//64,
            return_attn=self.return_attn,
            bidirectional=False,
            attn_mode=attn_mode,
            point_ffn_mode=self.point_ffn_mode,
            num_points=encoder_cfg.output_num,
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

        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pos_encoder = ManualPointWiseGemoFea(True, emb_dims)
        print(f"Using {self.knn_mode} knn mode")
        if self.point_augmented:
            print(f"Point-augmented mode enabled, point_ffn={self.point_ffn_mode}")

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
            anchor_embedding = self.emb_nn_anchor(anchor_points_dmean)
            if isinstance(action_embedding, tuple) or isinstance(anchor_embedding, tuple):
                action_embedding, pts = action_embedding
                act_down_sample = pts + action_points.mean(dim=2, keepdim=True)

                anchor_embedding, pts = anchor_embedding
                anch_down_sample = pts + anchor_points.mean(dim=2, keepdim=True)
            else:
                act_down_sample = action_points
                anch_down_sample = anchor_points
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

    def _backbone(self, act_query_emb, anch_query_emb,
                  cache_action_emb=None, cache_anch_emb=None,
                  act_index=None, anch_index=None,
                  action_points=None, anchor_points=None):
        act_query_emb = act_query_emb.transpose(2, 1).contiguous()
        anch_query_emb = anch_query_emb.transpose(2, 1).contiguous()
        act_src_emb = anch_query_emb if cache_anch_emb is None else cache_anch_emb
        anch_src_emb = act_query_emb if cache_action_emb is None else cache_action_emb
        if self.point_augmented:
            # mem_pts: (B, N, 3) in B,N,3 format for transformer
            mem_pts_action = None
            mem_pts_anchor = None
            if anchor_points is not None:
                # anchor_points is (B, 3, N) → need (B, M, 3)
                mem_pts_action = anchor_points.transpose(1, 2).contiguous()
            if action_points is not None:
                mem_pts_anchor = action_points.transpose(1, 2).contiguous()

            transformer_action_outputs = self.transformer_action(
                act_query_emb, act_src_emb, act_index,
                mem_pts=mem_pts_action,
            )
            transformer_anchor_outputs = self.transformer_anchor(
                anch_query_emb, anch_src_emb, anch_index,
                mem_pts=mem_pts_anchor,
            )
        else:
            transformer_action_outputs = self.transformer_action(
                act_query_emb, act_src_emb, act_index
            )
            transformer_anchor_outputs = self.transformer_anchor(
                anch_query_emb, anch_src_emb, anch_index
            )
        action_embedding_tf = transformer_action_outputs["src_embedding"]
        action_attn = transformer_action_outputs["src_attn"]
        anchor_embedding_tf = transformer_anchor_outputs["src_embedding"]
        anchor_attn = transformer_anchor_outputs["src_attn"]

        if not self.return_attn:
            action_attn = None
            anchor_attn = None

        # Extract corr_pts from point-augmented backbone
        action_corr_pts = transformer_action_outputs.get("src_corr_pts", None)
        anchor_corr_pts = transformer_anchor_outputs.get("src_corr_pts", None)

        del transformer_action_outputs, transformer_anchor_outputs
        return (action_embedding_tf, anchor_embedding_tf,
                action_attn, anchor_attn,
                action_corr_pts, anchor_corr_pts)

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
            anchor_attn,
            action_corr_pts=None,
            anchor_corr_pts=None,
            ):
        """"
        Coarse step of the network.
        Inputs are with shape of B, C, N
        """
        if self.point_augmented and action_corr_pts is not None:
            head_action_output = self.head_action(
                action_embedding_tf,
                action_embedding,
                anchor_embedding,
                action_points,
                anchor_points,
                act_down_sample,
                anch_down_sample,
                corr_points=action_corr_pts,
            )
        else:
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
        to_next_stage_corr_points = raw_pts + head_action_output["full_flow"][:, :3, :]
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
            if self.point_augmented and anchor_corr_pts is not None:
                head_anchor_output = self.head_anchor(
                    anchor_embedding_tf,
                    anchor_embedding,
                    action_embedding,
                    anchor_points,
                    action_points,
                    anch_down_sample,
                    act_down_sample,
                    corr_points=anchor_corr_pts,
                )
            else:
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
            to_next_anch_corr_points = raw_pts + head_anchor_output["full_flow"][:, :3, :]
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
                action_embedding, anchor_embedding,
                action_pt_pos, anchor_pt_pos,
                act_down_sample, anch_down_sample
            ) = self._embedding(*input)
            cache_act_emb = action_embedding
            cache_anch_emb = anchor_embedding
        else:
            (
                action_points, anchor_points,
                cache_act_emb, cache_anch_emb,
                action_pt_pos, anchor_pt_pos,
                act_down_sample, anch_down_sample,
                action_embedding, anchor_embedding   #  B, 3, N
            ) = input

        if self.knn_in_tf and self.knn_mode == 'emb':
            act_index = knn_query_with_embedding(action_embedding, K=8)
            anch_index = knn_query_with_embedding(anchor_embedding, K=8)
        elif self.knn_in_tf and self.knn_mode == 'pt':
            act_index = knn(act_down_sample, 8)
            anch_index = knn(anch_down_sample, 8)
        else:
            act_index = None
            anch_index = None
        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn,
            action_corr_pts,
            anchor_corr_pts,
        ) = self._backbone(
            action_embedding, anchor_embedding,
            cache_act_emb, cache_anch_emb,
            act_index, anch_index,
            action_points=act_down_sample if self.point_augmented else None,
            anchor_points=anch_down_sample if self.point_augmented else None
        )

        outputs = self.coarse_step(
            action_embedding_tf,
            anchor_embedding_tf,
            cache_act_emb,
            cache_anch_emb,
            action_points,
            anchor_points,
            act_down_sample,
            anch_down_sample,
            action_attn,
            anchor_attn,
            action_corr_pts=action_corr_pts,
            anchor_corr_pts=anchor_corr_pts,
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
            cache_act_emb, cache_anch_emb,
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
        knn_mode: str = "pt",
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
        self.knn_mode = knn_mode
        print(f"[TwoStageFlowTransformer] knn_mode: {knn_mode}")
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
        # Pop TF embedding, which will not be used at next stage
        act_emb_tf = shared_args.pop(4).detach().transpose(1, 2).contiguous()
        anch_emb_tf = shared_args.pop(4).detach().transpose(1, 2).contiguous()

        has_anchor = ("flow_anchor" in outputs
                      and outputs["flow_anchor"] is not None)
        down_sample = shared_args[0].shape[2] != shared_args[2].shape[2]

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
                          device=flow_action.device,
                          dtype=flow_action.dtype)
        h_b = torch.zeros(B, N, self.refine_hidden_dim,
                          device=flow_action.device,
                          dtype=flow_action.dtype) if has_anchor else None
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
                tgt_pt = key_point(
                    coarsed_act, tgt_pt,
                    mode=self.knn_mode,
                    emb_func=self._action_embedding)  # (B,N,k,3)
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
                    tgt_pt = key_point(
                        coarsed_anchor, tgt_pt,
                        mode=self.knn_mode,
                        emb_func=self._action_embedding)  # (B,N,k,3)
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

        shared_args[-2] = (action_bn3 + flow_action[:, :, :3]).permute(0, 2, 1).contiguous()
        if has_anchor:  # otherwise, shared_args[-1] is None
            shared_args[-1] = (anchor_bn3 + flow_anchor[:, :, :3]).permute(0, 2, 1).contiguous()
        return outputs

    @torch.no_grad()
    def inference(self, *input, **kwargs):
        return self.forward(*input, **kwargs)


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

