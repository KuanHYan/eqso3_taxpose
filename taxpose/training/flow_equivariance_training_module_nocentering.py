from typing import Any
import wandb
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.cuda.amp.autocast_mode import autocast
from pytorch3d.transforms import Transform3d

from taxpose.training.point_cloud_training_module import PointCloudTrainingModule
from taxpose.utils.color_utils import get_color
from taxpose.utils.se3 import (
    dense_flow_loss,
    dense_flow_distribution_loss,
    dualflow2pose,
    flow2pose,
    get_degree_angle,
    get_translation,
    PointCloudLoss,
    random_se3,
)

import matplotlib.cm as cm
import numpy as np
to_tensor = ToTensor()


class EquivarianceTrainingModule(PointCloudTrainingModule):
    def __init__(
        self,
        model=None,
        lr=1e-3,
        lr_cfg: dict = {
            "scheduler": "constant",
            "max_steps": 400,
            "warmup_ratio": 0.1,
            "min_lr": 1e-5,
            "by_epoch": True,
            "weight_decay": 1e-2,
        },
        image_log_period=500,
        action_weight=0.5,
        anchor_weight=0.5,
        displace_loss_weight=1,
        consistency_loss_weight=0.1,
        direct_correspondence_loss_weight=1,
        indirect_correspondence_loss_weight=1.0,
        res_smooth_loss_weight=1.0,
        start_res_flow_epoch=0,
        return_flow_component=False,
        weight_normalize="l1",
        sigmoid_on=False,
        softmax_temperature=None,
        flow_supervision="both",  # ('both', 'action2anchor', 'anchor2action')
        tr_super_time_ratio=0.0,
        point_cloud_loss: str = "MSE_SUM",
        debug=False,
        optimization_mode: str = "auto",
        # ── GPU 增强参数（已从数据集移至此处） ──
        action_rot_var: float = 3.1416,
        anchor_rot_var: float = 3.1416,
        trans_var: float = 0.5,
        action_rot_sample_method: str = "axis_angle",
        anchor_rot_sample_method: str = "axis_angle",
    ):
        super().__init__(
            model=model,
            lr=lr,
            image_log_period=image_log_period,
            debug=debug,
            optimization_mode=optimization_mode,
            **lr_cfg,
        )

        if '_' in point_cloud_loss:
            loss_mode = point_cloud_loss.split('_')
            pc_loss, reduc = loss_mode[0], str.lower(loss_mode[1])
        else:
            reduc = "sum"
            pc_loss = point_cloud_loss
        self.point_cloud_loss = PointCloudLoss(pc_loss, reduction=reduc)
        self.dense_flow_loss = PointCloudLoss(pc_loss, reduction=reduc)
        self.smooth_flow_loss = PointCloudLoss(pc_loss, reduction=reduc)
        self.model = model
        self.lr = lr
        self.image_log_period = image_log_period
        self.displace_loss_weight = displace_loss_weight
        self.action_weight = action_weight
        self.anchor_weight = anchor_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.direct_correspondence_loss_weight = direct_correspondence_loss_weight
        self.display_action = True
        self.display_anchor = True
        self.weight_normalize = weight_normalize

        self.return_flow_component = return_flow_component
        self.sigmoid_on = sigmoid_on
        self.softmax_temperature = softmax_temperature
        self.flow_supervision = flow_supervision
        if self.weight_normalize == "l1":
            assert self.sigmoid_on, "l1 weight normalization need sigmoid on"

        self.rotate_frobenius_loss_weight = 0.0
        self.tr_super_time_ratio = tr_super_time_ratio

        self.indirect_correspondence_loss_weight = indirect_correspondence_loss_weight
        self.res_smooth_loss_weight = res_smooth_loss_weight
        self.res_smooth_start_epoch = start_res_flow_epoch
        self.alpha = 1.0

        # GPU 增强参数
        self.action_rot_var = action_rot_var
        self.anchor_rot_var = anchor_rot_var
        self.trans_var = trans_var
        self.action_rot_sample_method = action_rot_sample_method
        self.anchor_rot_sample_method = anchor_rot_sample_method

    # def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
    #     # 仅加载 self.model 在 state_dict 有的参数
    #     model_dict = self.state_dict()
    #     pretrained_dict = {k: v for k, v in state_dict.items()
    #                if k in model_dict and v.shape == model_dict[k].shape}
    #     model_dict.update(pretrained_dict)
    #     super().load_state_dict(
    #         model_dict,
    #         strict=strict,
    #         assign=assign,
    #     )
    #     return

    def compute_loss(self, model_output, batch, log_values={}, loss_prefix=""):
        x_action = model_output["flow_action"]
        x_anchor = model_output["flow_anchor"]

        # The original point clouds from the demonstration.
        points_action = batch["points_action"]  # action point clouds
        points_anchor = batch["points_anchor"]  # anchor point clouds
        # The point clouds transformed by the ground truth transforms.
        points_trans_action = batch["points_action_trans"]  # T0 -> action point clouds
        points_trans_anchor = batch["points_anchor_trans"]  # T1 -> anchor point clouds
        input_act_pts = points_trans_action
        input_anch_pts = points_trans_anchor

        # If we've applied some sampling, we need to extract the predictions too...
        if "sampled_ixs_action" in model_output:
            ixs_action = model_output["sampled_ixs_action"].unsqueeze(-1)
            points_action = torch.take_along_dim(points_action, ixs_action, dim=1)
            points_trans_action = torch.take_along_dim(
                points_trans_action, ixs_action, dim=1
            )

        if "sampled_ixs_anchor" in model_output:
            ixs_anchor = model_output["sampled_ixs_anchor"].unsqueeze(-1)
            points_anchor = torch.take_along_dim(points_anchor, ixs_anchor, dim=1)
            points_trans_anchor = torch.take_along_dim(
                points_trans_anchor, ixs_anchor, dim=1
            )

        # Get the transforms.
        T0 = Transform3d(matrix=batch["T0"])
        T1 = Transform3d(matrix=batch["T1"])

        # rotation component applied to points_action
        R0_max, R0_min, R0_mean = get_degree_angle(T0)
        # rotation component applied to points_anchor
        R1_max, R1_min, R1_mean = get_degree_angle(T1)
        # translation component applied to points_action
        t0_max, t0_min, t0_mean = get_translation(T0)
        # translation component applied to points_anchor
        t1_max, t1_min, t1_mean = get_translation(T1)

        # Extract predictred flow and weight
        pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
        pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)

        if points_trans_action.shape[1] != pred_flow_action.shape[1]:
            input_act_pts = model_output["act_down_sample"]
            input_anch_pts = model_output["anch_down_sample"]

        if self.flow_supervision == "both":
            pred_T_action = dualflow2pose(
                xyz_src=input_act_pts,
                xyz_tgt=input_anch_pts,
                flow_src=pred_flow_action,
                flow_tgt=pred_flow_anchor,
                weights_src=pred_w_action,
                weights_tgt=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature,
                training=True  # self.training,
            )

            error_R_max, error_R_min, error_R_mean = get_degree_angle(
                T0.inverse().compose(T1).compose(pred_T_action.inverse())
            )
            error_t_max, error_t_min, error_t_mean = get_translation(
                T0.inverse().compose(T1).compose(pred_T_action.inverse())
            )
            if "hook_trans_T" in model_output:
                hook_trans_T = model_output["hook_trans_T"]
                for i, Ti in enumerate(hook_trans_T):
                    error_R_i = get_degree_angle(
                        T0.inverse().compose(T1).compose(Ti.inverse())
                    )[-1]
                    error_t_i = get_translation(
                        T0.inverse().compose(T1).compose(Ti.inverse())
                    )[-1]
                    log_values[f"{loss_prefix}error_R_stage{i}"] = error_R_i
                    log_values[f"{loss_prefix}error_t_stage{i}"] = error_t_i
            if "observing_deltaT" in model_output:
                observing_deltaT = model_output["observing_deltaT"]
                for i, (theta_i, delta_t_i) in enumerate(observing_deltaT):
                    log_values[f"{loss_prefix}delta_theta_stage{i}"] = (theta_i / torch.pi * 180.0).mean()
                    log_values[f"{loss_prefix}delta_t_stage{i}"] = (delta_t_i).mean()
            # Loss associated with ground truth transform
            pred_points_action = pred_T_action.transform_points(points_trans_action)
            points_action_target = T1.transform_points(points_action)
            point_loss_action = self.point_cloud_loss(
                pred_points_action, points_action_target,
            )

            # ##
            # pa = points_action_target.detach()
            # pb = points_trans_anchor.detach()
            # matrix_pb_pbt = pb @ pb.transpose(1,2)
            # w_a2b = pa @ pb.transpose(1,2) @ matrix_pb_pbt.inverse()
            # w_a2b_row = w_a2b.sum(dim=2, keepdim=True)
            # print(f"debug row sum: {w_a2b_row[0, 0:10]}")

            # Loss associated flow vectors matching a consistent rigid transform
            induced_flow_action = (
                pred_T_action.transform_points(input_act_pts) - input_act_pts
            ).detach()
            smoothness_loss_action = self.smooth_flow_loss(
                pred_flow_action, induced_flow_action,
            )
            # loss associated with dense flow
            # pred_T_action=T1T0^-1
            gt_T_action = T0.inverse().compose(T1)
            dense_loss_action = self.dense_flow_loss(
                pred_flow_action,
                gt_T_action.transform_points(input_act_pts) - input_act_pts,
            )
            # dense_loss_action = dense_flow_distribution_loss(
            #     points=input_act_pts,
            #     flow_pred=pred_flow_action,
            #     variance_pred=model_output["residual_flow_action"],
            #     trans_gt=gt_T_action,
            # )
            pred_T_anchor = pred_T_action.inverse()
            # Loss associated with ground truth transform
            pred_points_anchor = pred_T_anchor.transform_points(points_trans_anchor)
            points_anchor_target = T0.transform_points(points_anchor)
            point_loss_anchor = self.point_cloud_loss(
                pred_points_anchor,
                points_anchor_target,
            )

            # Loss associated flow vectors matching a consistent rigid transform
            induced_flow_anchor = (
                pred_T_anchor.transform_points(input_anch_pts) - input_anch_pts
            ).detach()
            smoothness_loss_anchor = self.smooth_flow_loss(
                pred_flow_anchor,
                induced_flow_anchor,
            )

            # loss associated with dense flow
            # pred_T_action=T1T0^-1
            gt_T_anchor = T1.inverse().compose(T0)
            dense_loss_anchor = self.dense_flow_loss(
                pred_flow_anchor,
                gt_T_anchor.transform_points(input_anch_pts) - input_anch_pts,
            )
            # dense_loss_anchor = dense_flow_distribution_loss(
            #     points=inputs_anch_pts,
            #     flow_pred=pred_flow_anchor,
            #     variance_pred=model_output["residual_flow_anchor"],
            #     trans_gt=gt_T_anchor,
            # )

            # self.action_weight = (self.action_weight) / (
            #     self.action_weight + self.anchor_weight
            # )
            # self.anchor_weight = (self.anchor_weight) / (
            #     self.action_weight + self.anchor_weight
            # )

        elif self.flow_supervision == "action2anchor":
            pred_T_action = flow2pose(
                xyz=points_trans_action,
                flow=pred_flow_action,
                weights=pred_w_action,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature,
            )
            induced_flow_action = (
                pred_T_action.transform_points(points_trans_action)
                - points_trans_action
            ).detach()
            pred_points_action = pred_T_action.transform_points(points_trans_action)

            # pred_T_action=T1T0^-1
            gt_T_action = T0.inverse().compose(T1)
            points_action_target = T1.transform_points(points_action)

            error_R_max, error_R_min, error_R_mean = get_degree_angle(
                T0.inverse().compose(T1).compose(pred_T_action.inverse())
            )

            error_t_max, error_t_min, error_t_mean = get_translation(
                T0.inverse().compose(T1).compose(pred_T_action.inverse())
            )

            # Loss associated with ground truth transform
            point_loss_action = self.point_cloud_loss(
                pred_points_action,
                points_action_target,
            )

            # Loss associated flow vectors matching a consistent rigid transform
            smoothness_loss_action = self.smooth_flow_loss(
                pred_flow_action,
                induced_flow_action,
            )

            dense_loss_action = self.dense_flow_loss(
                pred_flow_action,
                gt_T_action.transform_points(points_trans_action) - points_trans_action,
            )

            # Zero anchor terms
            self.anchor_weight = 0
            self.action_weight = (self.action_weight) / (
                self.action_weight + self.anchor_weight
            )
            point_loss_anchor = 0
            smoothness_loss_anchor = 0
            dense_loss_anchor = 0
        elif self.flow_supervision == "anchor2action":
            pred_T_anchor = flow2pose(
                xyz=points_trans_anchor,
                flow=pred_flow_anchor,
                weights=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature,
            )
            induced_flow_anchor = (
                pred_T_anchor.transform_points(points_trans_anchor)
                - points_trans_anchor
            ).detach()
            pred_points_anchor = pred_T_anchor.transform_points(points_trans_anchor)

            # pred_T_action=T1T0^-1
            gt_T_anchor = T1.inverse().compose(T0)
            points_anchor_target = T0.transform_points(points_anchor)

            error_R_max, error_R_min, error_R_mean = get_degree_angle(
                T1.inverse().compose(T0).compose(pred_T_anchor.inverse())
            )

            error_t_max, error_t_min, error_t_mean = get_translation(
                T1.inverse().compose(T0).compose(pred_T_anchor.inverse())
            )

            # Loss associated with ground truth transform
            point_loss_anchor = self.point_cloud_loss(
                pred_points_anchor,
                points_anchor_target,
            )

            # Loss associated flow vectors matching a consistent rigid transform
            smoothness_loss_anchor = self.smooth_flow_loss(
                pred_flow_anchor,
                induced_flow_anchor,
            )
            dense_loss_anchor = self.dense_flow_loss(
                pred_flow_anchor,
                gt_T_anchor.transform_points(points_trans_anchor) - points_trans_anchor,
            )
            # Zero action terms
            self.action_weight = 0
            self.anchor_weight = (self.anchor_weight) / (
                self.action_weight + self.anchor_weight
            )
            point_loss_action = 0
            smoothness_loss_action = 0
            dense_loss_action = 0

        point_loss = (
            self.action_weight * point_loss_action
            + self.anchor_weight * point_loss_anchor
        )

        dense_loss = (
            self.action_weight * dense_loss_action
            + self.anchor_weight * dense_loss_anchor
        )

        smoothness_loss = (
            self.action_weight * smoothness_loss_action
            + self.anchor_weight * smoothness_loss_anchor
        )

        if model_output.get("refined_loss", None) is not None:
            # TODO: Consider using a weight beta for the refined loss
            refined_loss = sum(model_output["refined_loss"])
            log_values[loss_prefix + "refined_loss"] = refined_loss.detach()
            loss = (refined_loss,)
        else:
            loss = (
                self.displace_loss_weight * point_loss,
                self.consistency_loss_weight * smoothness_loss,
                self.direct_correspondence_loss_weight * dense_loss,
            )

        corr_std_act = model_output.get("corr_std_act", None)
        corr_std_anch = model_output.get("corr_std_anch", None)
        if corr_std_act is not None and corr_std_anch is not None:
            std_loss = (
                0.5 * self.point_cloud_loss(corr_std_act, T0.inverse().transform_points(input_act_pts)) +
                0.5 * self.point_cloud_loss(corr_std_anch, T1.inverse().transform_points(input_anch_pts))
            )
            loss += (std_loss,)
            log_values[loss_prefix + "std_loss"] = std_loss.detach()

        if "tuned_T" in model_output:
            # 使用MSE直接计算预测变换和真值变换的误差
            pred_T = model_output["tuned_T"]
            gt_T = gt_T_action
            loss += (F.mse_loss(pred_T.get_matrix(), gt_T.get_matrix(), reduction="mean"))

        if model_output.get("coarse_loss", None) is not None:
            coarse_loss = sum(model_output["coarse_loss"])
            log_values[loss_prefix + "coarse_loss"] = coarse_loss.detach()
            loss += (self.alpha * coarse_loss,)

        if self.rotate_frobenius_loss_weight > 0.0:
            R_delta = gt_T_action.get_matrix() - pred_T_action.get_matrix()
            loss_tr = (
                self.rotate_frobenius_loss_weight
                * (R_delta**2).sum(dim=(-1, -2)).mean()
            )
            log_values[loss_prefix + "tr_loss"] = loss_tr.detach()
            loss += (loss_tr,)

        if self.indirect_correspondence_loss_weight > 0.0:
            with torch.no_grad():
                act_perm = torch.randperm(input_act_pts.shape[1])
                action_points = input_act_pts[:, act_perm, :]

                anch_perm = torch.randperm(input_anch_pts.shape[1])
                anchor_points = input_anch_pts[:, anch_perm, :]

                shuffled_output = self.model(
                    action_points,
                    anchor_points,
                )
                inv_perm_a = torch.argsort(act_perm)
                inv_perm_b = torch.argsort(anch_perm)
                shuffled_act_corr = shuffled_output["flow_action"][:, :, :3][
                    :, inv_perm_a
                ]
                shuffled_anch_corr = shuffled_output["flow_anchor"][:, :, :3][
                    :, inv_perm_b
                ]
            ww = self.indirect_correspondence_loss_weight
            loss_indirect = ww * self.point_cloud_loss(
                pred_flow_action, shuffled_act_corr
            ) + ww * self.point_cloud_loss(pred_flow_anchor, shuffled_anch_corr)
            loss += (loss_indirect,)
            log_values[loss_prefix + "indirect_loss"] = loss_indirect.detach()
            del shuffled_output, shuffled_act_corr, shuffled_anch_corr
            del act_perm, anch_perm, inv_perm_a, inv_perm_b

        if (
            self.res_smooth_loss_weight > 0.0
            and self.current_epoch >= self.res_smooth_start_epoch
        ):
            act_res_smooth_loss = self.compute_res_smoothness_loss(
                model_output["residual_flow_action"], input_act_pts
            )
            anch_res_smooth_loss = self.compute_res_smoothness_loss(
                model_output["residual_flow_anchor"], input_anch_pts
            )
            loss_res = (
                self.res_smooth_loss_weight * act_res_smooth_loss
                + self.res_smooth_loss_weight * anch_res_smooth_loss
            )
            loss += (loss_res,)
            log_values[loss_prefix + "res_smooth_loss"] = loss_res.detach()

        if "act_emb_similarity" in model_output:
            loss += (self.compute_emb_sim_loss(model_output, log_values, loss_prefix),)

        log_values[loss_prefix + "point_loss"] = self.displace_loss_weight * point_loss.detach()
        log_values[loss_prefix + "smoothness_loss"] = (
            self.consistency_loss_weight * smoothness_loss.detach()
        )
        log_values[loss_prefix + "dense_loss"] = (
            self.direct_correspondence_loss_weight * dense_loss.detach()
        )
        # centered_pred_ps_A = pred_points_action.detach()
        # centered_pred_ps_A = centered_pred_ps_A - centered_pred_ps_A.mean(
        #     dim=1, keepdim=True
        # )
        # centered_gt_ps = points_action_target.detach()
        # centered_gt_ps = centered_gt_ps - centered_gt_ps.mean(dim=1, keepdim=True)
        # log_values[loss_prefix + "only_Rotate_L2_pcs_distance"] = \
        #     mse_criterion(centered_pred_ps_A, centered_gt_ps)
        # log_values[loss_prefix + "R0_mean"] = R0_mean
        # log_values[loss_prefix + "R0_max"] = R0_max
        # log_values[loss_prefix + "R0_min"] = R0_min
        # log_values[loss_prefix + "R1_mean"] = R1_mean
        # log_values[loss_prefix + "R1_max"] = R1_max
        # log_values[loss_prefix + "R1_min"] = R1_min

        # log_values[loss_prefix + "t0_mean"] = t0_mean
        # log_values[loss_prefix + "t0_max"] = t0_max
        # log_values[loss_prefix + "t0_min"] = t0_min
        # log_values[loss_prefix + "t1_mean"] = t1_mean
        # log_values[loss_prefix + "t1_max"] = t1_max
        # log_values[loss_prefix + "t1_min"] = t1_min

        log_values[loss_prefix + "error_R_mean"] = error_R_mean
        log_values[loss_prefix + "error_t_mean"] = error_t_mean

        return loss, log_values

    def extract_flow_and_weight(self, x):
        # x: Batch, num_points, 4
        pred_flow = x[:, :, :3]
        if x.shape[2] > 3:
            if self.sigmoid_on:
                pred_w = torch.sigmoid(x[:, :, 3])
            else:
                pred_w = x[:, :, 3]
        else:
            pred_w = None
        return pred_flow, pred_w

    def compute_emb_sim_loss(self, model_output, log_values, loss_prefix):
        anch_emb_sim = model_output["anch_emb_similarity"]
        act_emb_sim = model_output["act_emb_similarity"]
        loss = (1 - act_emb_sim).mean() * 0.5 + (1 - anch_emb_sim).mean() * 0.5
        log_values[loss_prefix + "emb_sim_loss"] = loss
        log_values[loss_prefix + "emb_sim"] = act_emb_sim.detach().mean()
        return loss

    def compute_res_smoothness_loss(self, pred_res_flow, points_trans_action, k=16):
        b, n, c = pred_res_flow.shape
        device = pred_res_flow.device
        with torch.no_grad():
            pts = points_trans_action
            pair_dis = torch.cdist(pts, pts)
            top_k_idx = pair_dis.topk(k, largest=False)[1]
            batch_offset = torch.arange(b, device=device).view(b, 1, 1) * n
            idx = top_k_idx + batch_offset  # (B, M, k)
            idx_flat = idx.reshape(-1, k)
            delta = pred_res_flow.detach()
            gathered_feat = delta.reshape(-1, c)[idx_flat, :].view(b, n, k, -1)

        loss_smooth = self.smooth_flow_loss(
            pred_res_flow.unsqueeze(2).expand(-1, -1, k, -1), gathered_feat
        )
        return loss_smooth

    def module_step(self, batch, batch_idx):
        # ── GPU 增强：随机旋转 + 平移（原先在 Dataset.__getitem__ 的 CPU 上执行）──
        B = batch["points_action"].shape[0]
        device = batch["points_action"].device

        T0 = random_se3(
            B,
            rot_var=self.action_rot_var,
            trans_var=self.trans_var,
            device=device,
            rot_sample_method=self.action_rot_sample_method,
        )
        T1 = random_se3(
            B,
            rot_var=self.anchor_rot_var,
            trans_var=self.trans_var,
            device=device,
            rot_sample_method=self.anchor_rot_sample_method,
        )

        batch["points_action_trans"] = T0.transform_points(batch["points_action"])
        batch["points_anchor_trans"] = T1.transform_points(batch["points_anchor"])
        batch["T0"] = T0.get_matrix()
        batch["T1"] = T1.get_matrix()

        # ── 原有逻辑不变 ──
        points_trans_action = batch["points_action_trans"]
        points_trans_anchor = batch["points_anchor_trans"]
        action_features = (
            batch["action_features"] if "action_features" in batch else None
        )
        anchor_features = (
            batch["anchor_features"] if "anchor_features" in batch else None
        )

        if "phase_onehot" in batch:
            phase_onehot = batch["phase_onehot"]
        else:
            phase_onehot = None

        forward_fun = self.model.forward if self.train else self.model.inference
        compute_loss = partial(self.compute_loss, batch=batch) if self.train else None

        model_output = forward_fun(
            points_trans_action,
            points_trans_anchor,
            action_features,
            anchor_features,
            phase_onehot,
            compute_loss=compute_loss,
        )

        log_values = {}
        loss, log_values = self.compute_loss(
            model_output, batch, log_values=log_values, loss_prefix=""
        )
        return loss, log_values

    def on_train_epoch_start(self) -> None:
        if (
            self.current_epoch >= self.tr_super_time_ratio * self.end_lr_steps
            and self.rotate_frobenius_loss_weight == 0.0
        ):
            self.rotate_frobenius_loss_weight = 1.0
            self.print("rotate_frobenius_loss_weight = 1.0")
        # progress = self.current_epoch / self.trainer.max_epochs
        self.alpha = 1.0  # max(0.1, 1.0 - progress)  # 1.0 → 0.1
        return super().on_train_epoch_start()

    def forward(
        self,
        points_trans_action,
        points_trans_anchor,
        action_features,
        anchor_features,
        phase_onehot=None,
    ) -> Any:
        model_output = self.model(
            points_trans_action,
            points_trans_anchor,
            action_features,
            anchor_features,
            phase_onehot,
        )

        # If we've applied some sampling, we need to extract the predictions too...
        if "sampled_ixs_action" in model_output:
            ixs_action = model_output["sampled_ixs_action"].unsqueeze(-1)
            points_trans_action = torch.take_along_dim(
                points_trans_action, ixs_action, dim=1
            )

        if "sampled_ixs_anchor" in model_output:
            ixs_anchor = model_output["sampled_ixs_anchor"].unsqueeze(-1)
            points_trans_anchor = torch.take_along_dim(
                points_trans_anchor, ixs_anchor, dim=1
            )

        x_action = model_output["flow_action"]
        x_anchor = model_output["flow_anchor"]

        pred_flow_action = x_action[:, :, :3]
        if x_action.shape[2] > 3:
            if self.sigmoid_on:
                pred_w_action = torch.sigmoid(x_action[:, :, 3])
            else:
                pred_w_action = x_action[:, :, 3]
        else:
            pred_w_action = None

        pred_flow_anchor = x_anchor[:, :, :3]
        if x_anchor.shape[2] > 3:
            if self.sigmoid_on:
                pred_w_anchor = torch.sigmoid(x_anchor[:, :, 3])
            else:
                pred_w_anchor = x_anchor[:, :, 3]
        else:
            pred_w_anchor = None

        pred_T_action = dualflow2pose(
            xyz_src=points_trans_action,
            xyz_tgt=points_trans_anchor,
            flow_src=pred_flow_action,
            flow_tgt=pred_flow_anchor,
            weights_src=pred_w_action,
            weights_tgt=pred_w_anchor,
            return_transform3d=True,
            normalization_scehme=self.weight_normalize,
            temperature=self.softmax_temperature,
            training=self.training,
        )

        pred_points_action = pred_T_action.transform_points(points_trans_action)

        res = {
            "pred_points_action": pred_points_action,
            "pred_flow_action": pred_flow_action,
            "pred_w_action": pred_w_action,
            "pred_T_action": pred_T_action,
        }

        if "sampled_ixs_action" in model_output:
            res["sampled_ixs_action"] = model_output["sampled_ixs_action"]

        return res

    @torch.no_grad()
    @autocast(enabled=False)
    def visualize_results(self, batch, batch_idx):
        # classes = batch['classes']
        # points = batch['points']
        self.model.eval()
        points_action = batch["points_action"]
        points_anchor = batch["points_anchor"]
        # points_trans = batch['points_trans']
        points_trans_action = batch["points_action_trans"]
        input_act_pts = points_trans_action
        points_trans_anchor = batch["points_anchor_trans"]
        inputs_anch_pts = points_trans_anchor
        action_features = (
            batch["action_features"] if "action_features" in batch else None
        )
        anchor_features = (
            batch["anchor_features"] if "anchor_features" in batch else None
        )
        action_symmetry_rgb = (
            batch["action_symmetry_rgb"] if "action_symmetry_rgb" in batch else None
        )
        anchor_symmetry_rgb = (
            batch["anchor_symmetry_rgb"] if "anchor_symmetry_rgb" in batch else None
        )
        onehot = batch["phase_onehot"] if "phase_onehot" in batch else None

        T0 = Transform3d(matrix=batch["T0"])
        T1 = Transform3d(matrix=batch["T1"])

        model_output = self.model.inference(
            points_trans_action,
            points_trans_anchor,
            action_features,
            anchor_features,
            onehot,
        )
        x_action = model_output["flow_action"]
        x_anchor = model_output["flow_anchor"]
        if x_action.shape[1] != points_trans_action.shape[1]:
            input_act_pts = model_output["act_down_sample"]
            inputs_anch_pts = model_output["anch_down_sample"]

        # If we've applied some sampling, we need to extract the predictions too...
        if "sampled_ixs_action" in model_output:
            ixs_action = model_output["sampled_ixs_action"].unsqueeze(-1)
            points_action = torch.take_along_dim(points_action, ixs_action, dim=1)
            points_trans_action = torch.take_along_dim(
                points_trans_action, ixs_action, dim=1
            )
            if action_symmetry_rgb is not None:
                action_symmetry_rgb = torch.take_along_dim(
                    action_symmetry_rgb, ixs_action, dim=1
                )
                action_symmetry_features = torch.take_along_dim(
                    action_symmetry_features, ixs_action, dim=1
                )

        if "sampled_ixs_anchor" in model_output:
            ixs_anchor = model_output["sampled_ixs_anchor"].unsqueeze(-1)
            points_anchor = torch.take_along_dim(points_anchor, ixs_anchor, dim=1)
            points_trans_anchor = torch.take_along_dim(
                points_trans_anchor, ixs_anchor, dim=1
            )
            if anchor_symmetry_rgb is not None:
                anchor_symmetry_rgb = torch.take_along_dim(
                    anchor_symmetry_rgb, ixs_anchor, dim=1
                )
                anchor_symmetry_features = torch.take_along_dim(
                    anchor_symmetry_features, ixs_anchor, dim=1
                )

        pred_flow_action = x_action[:, :, :3]
        if x_action.shape[2] > 3:
            if self.sigmoid_on:
                pred_w_action = torch.sigmoid(x_action[:, :, 3])
            else:
                pred_w_action = x_action[:, :, 3]
        else:
            pred_w_action = None

        pred_flow_anchor = x_anchor[:, :, :3]
        if x_anchor.shape[2] > 3:
            if self.sigmoid_on:
                pred_w_anchor = torch.sigmoid(x_anchor[:, :, 3])
            else:
                pred_w_anchor = x_anchor[:, :, 3]
        else:
            pred_w_anchor = None
        if self.flow_supervision == "both":
            pred_T_action = dualflow2pose(
                xyz_src=input_act_pts,
                xyz_tgt=inputs_anch_pts,
                flow_src=pred_flow_action,
                flow_tgt=pred_flow_anchor,
                weights_src=pred_w_action,
                weights_tgt=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature,
            )
        elif self.flow_supervision == "action2anchor":
            pred_T_action = flow2pose(
                xyz=points_trans_action,
                flow=pred_flow_action,
                weights=pred_w_action,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature,
            )
            pred_T_anchor = pred_T_action.inverse()

        elif self.flow_supervision == "anchor2action":
            pred_T_anchor = flow2pose(
                xyz=points_trans_anchor,
                flow=pred_flow_anchor,
                weights=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature,
            )
            pred_T_action = pred_T_anchor.inverse()

        pred_points_action = pred_T_action.transform_points(points_trans_action)
        points_action_target = T1.transform_points(points_action)

        res_images = {}

        demo_points = get_color(
            tensor_list=[points_action[0], points_anchor[0]], color_list=["blue", "red"]
        )
        res_images["demo_points"] = wandb.Object3D(demo_points)

        action_transformed_action = get_color(
            tensor_list=[points_action[0], points_trans_action[0]],
            color_list=["blue", "red"],
        )
        res_images["action_transformed_action"] = wandb.Object3D(
            action_transformed_action
        )

        anchor_transformed_anchor = get_color(
            tensor_list=[points_anchor[0], points_trans_anchor[0]],
            color_list=["blue", "red"],
        )
        res_images["anchor_transformed_anchor"] = wandb.Object3D(
            anchor_transformed_anchor
        )

        transformed_input_points = get_color(
            tensor_list=[points_trans_action[0], points_trans_anchor[0]],
            color_list=["blue", "red"],
        )
        res_images["transformed_input_points"] = wandb.Object3D(
            transformed_input_points
        )

        demo_points_apply_action_transform = get_color(
            tensor_list=[pred_points_action[0], points_trans_anchor[0]],
            color_list=["blue", "red"],
        )
        res_images["demo_points_apply_action_transform"] = wandb.Object3D(
            demo_points_apply_action_transform
        )

        apply_action_transform_demo_comparable = get_color(
            tensor_list=[
                T1.inverse().transform_points(pred_points_action)[0],
                T1.inverse().transform_points(points_trans_anchor)[0],
            ],
            color_list=["blue", "red"],
        )
        res_images["apply_action_transform_demo_comparable"] = wandb.Object3D(
            apply_action_transform_demo_comparable
        )

        predicted_vs_gt_transform_applied = get_color(
            tensor_list=[
                T1.inverse().transform_points(pred_points_action)[0],
                points_action[0],
                T1.inverse().transform_points(points_trans_anchor)[0],
            ],
            color_list=[
                "blue",
                "green",
                "red",
            ],
        )
        res_images["predicted_vs_gt_transform_applied"] = wandb.Object3D(
            predicted_vs_gt_transform_applied
        )

        apply_predicted_transform = get_color(
            tensor_list=[
                T1.inverse().transform_points(pred_points_action)[0],
                T1.inverse().transform_points(points_trans_action)[0],
                T1.inverse().transform_points(points_trans_anchor)[0],
            ],
            color_list=[
                "blue",
                "orange",
                "red",
            ],
        )
        res_images["apply_predicted_transform"] = wandb.Object3D(
            apply_predicted_transform
        )

        loss_points_action = get_color(
            tensor_list=[points_action_target[0], pred_points_action[0]],
            color_list=["green", "red"],
        )
        res_images["loss_points_action"] = wandb.Object3D(loss_points_action)

        # Stack points and colors
        if action_symmetry_rgb is not None:
            action_xyzrgb = torch.cat([points_action[0], action_symmetry_rgb[0]], dim=1)
            anchor_xyzrgb = torch.cat([points_anchor[0], anchor_symmetry_rgb[0]], dim=1)
            xyzrgb = torch.cat([action_xyzrgb, anchor_xyzrgb], dim=0)
            res_images["symmetry_vis"] = wandb.Object3D(xyzrgb.cpu().numpy())

        if "attns" in model_output and model_output["attns"] is not None:
            batch_0_attn = model_output["attns"][0].detach().cpu()
            attns_for_anchor = batch_0_attn.sum(dim=0)  # [N, N] -> N, 1
            # mean_entropy = -torch.sum(batch_0_attn * torch.log(batch_0_attn + 1e-8), dim=-1)
            # idx = torch.randint(0, len(attns_for_anchor), (40,)).to(device=batch_0_attn.device)
            w_min = torch.quantile(attns_for_anchor, 0.01)
            w_max = torch.quantile(attns_for_anchor, 0.99)
            attns_for_anchor = torch.clamp(attns_for_anchor, w_min, w_max)
            w_norm = (attns_for_anchor - w_min) / (w_max - w_min + 1e-8)

            # 使用 coolwarm (蓝-白-红) 映射
            color_dist = (
                255 * cm.coolwarm(w_norm.cpu().numpy())[:, :3]
            )  # [N, 3], ignore alpha
            vis_pts = points_trans_anchor[0].detach().cpu()
            if vis_pts.shape[0] != color_dist.shape[0]:
                vis_pts = model_output["anch_down_sample"][0].detach().cpu()
            attns_action = np.concatenate([vis_pts.numpy(), color_dist], axis=1)
            res_images["attns_action"] = wandb.Object3D(attns_action)

        corr_points = model_output["corr_points_action"][0].cpu()
        corr_points = corr_points - corr_points.mean(dim=0, keepdim=True)
        input_act_pts_target = T0.inverse().compose(T1).transform_points(input_act_pts)
        maybe_corr = pred_flow_action[0].cpu() + input_act_pts[0].cpu()
        maybe_corr = maybe_corr - maybe_corr.mean(dim=0, keepdim=True)
        draw_points = get_color(
            tensor_list=[
                input_act_pts_target[0].cpu() - input_act_pts_target[0].cpu().mean(dim=0, keepdim=True),
                corr_points,
                maybe_corr,
            ],
            color_list=["blue", "red", "green"],
        )
        res_images["corr_points_action"] = wandb.Object3D(draw_points)
        res_images.update(
            EquivarianceTrainingModule._make_weight_visualizations(
                model=self.model,
                batch=batch,
                idx=0,
                model_output_cache={
                    "pred_flow_action": pred_flow_action,
                    "pred_w_action": pred_w_action,
                    "input_points": input_act_pts,
                    "gt_points_action": input_act_pts_target,
                }))
        return res_images


    # ─── Weight 可视化 (人工分析逐点权重质量) ───

    @torch.no_grad()
    @staticmethod
    def _make_weight_visualizations(model, batch, idx,
                                    max_samples: int = 4,
                                    model_output_cache=None):
        """生成逐点权重的多种可视化, 返回 wandb 日志字典.

        可视化方案:
          1. weight_colored_pts  — 点云按权重着色 (红=高权, 蓝=低权)
          2. weight_vs_error      — 权重 vs flow 误差散点图 (检验相关性)
          3. weight_histogram     — 权重分布直方图
          4. topk_weighted_pts   — 高亮 Top-K 权重点的 flow 向量
        """
        model.eval()
        T0 = Transform3d(matrix=batch["T0"])
        T1 = Transform3d(matrix=batch["T1"])
        gt_T_action = T0.inverse().compose(T1)  # (B, N, 3)

        res = {}

        # ── 确定性前向, 拿到逐点 weight ──
        if model_output_cache is None:
            pts_trans_action = batch["points_action_trans"]  # (B, N, 3)
            pts_trans_anchor = batch["points_anchor_trans"]  # (B, N, 3)
            B = min(pts_trans_action.shape[0], max_samples)
            b = random.randint(0, B-1)
            model_output = model(
                pts_trans_action[b:b+1], pts_trans_anchor[b:b+1])
            flow_act = model_output["flow_action"]              # (1, N, 4)
            pred_flow = flow_act[0, :, :3]                      # (N, 3)
            pred_w = torch.sigmoid(flow_act[0, :, 3])           # (N,)
            # GT 刚性流: gt_T(p) - p  (只算当前样本)
            point_target_b = gt_T_action.transform_points(
                pts_trans_action[b:b+1])[0]                      # (N, 3)
            gt_flow = point_target_b - pts_trans_action[b]       # (N, 3)
        else:
            pts_trans_action = model_output_cache["input_points"]
            b = idx
            pred_flow = model_output_cache["pred_flow_action"][b]  # (N, 3)
            # 注意: cache 中的 weight 可能已经 sigmoid 过,
            # 统一在此确保 [0,1] 范围
            w_raw = model_output_cache["pred_w_action"][b]
            pred_w = w_raw if ((w_raw >= 0).all() and (w_raw <= 1).all()) \
                     else torch.sigmoid(w_raw)
            point_target_b = model_output_cache["gt_points_action"][b]  # (N, 3)
            gt_flow = point_target_b - pts_trans_action[b]              # (N, 3)

        # 逐点 flow 误差
        flow_err = (pred_flow - gt_flow).norm(p=2, dim=-1)   # (N,)

        pts = pts_trans_action[b].cpu().numpy()              # (N, 3)
        w = pred_w.cpu().numpy()
        ferr = flow_err.cpu().numpy()
        pred_f = pred_flow.cpu().numpy()

        # ── 1. 按权重着色的点云 ──
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
        # colors = np.stack([w_norm, 0.2, 1 - w_norm], axis=-1) * 255  # 红→蓝
        color_dist = (
            255 * cm.coolwarm(w_norm)[:, :3]
        )
        pts_rgb = np.concatenate([pts, color_dist], axis=1)
        res["weight_colored_pts"] = wandb.Object3D(pts_rgb)

        # ── 2. 权重 vs flow 误差散点图 ──
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # 散点
        axes[0].scatter(w, ferr, c=w_norm, cmap='coolwarm',
                        alpha=0.5, s=10, edgecolors='none')
        axes[0].set_xlabel("Predicted Weight")
        axes[0].set_ylabel("Flow Error (L2)")
        axes[0].set_title("Weight vs Flow Error")
        # 分箱统计 (binned mean ± std)
        bins = np.linspace(0, 1, 11)
        bin_idx = np.digitize(w, bins) - 1
        bin_mean = [ferr[bin_idx == i].mean() if (bin_idx == i).sum() > 0
                    else np.nan for i in range(10)]
        bin_std = [ferr[bin_idx == i].std() if (bin_idx == i).sum() > 0
                    else np.nan for i in range(10)]
        bin_c = (bins[:-1] + bins[1:]) / 2
        axes[1].bar(bin_c, bin_mean, width=0.08, yerr=bin_std,
                    color='steelblue', alpha=0.7, capsize=3)
        axes[1].set_xlabel("Predicted Weight (binned)")
        axes[1].set_ylabel("Mean Flow Error")
        axes[1].set_title("Binned Weight vs Mean Error")
        plt.tight_layout()
        res["weight_vs_error"] = wandb.Image(fig)
        plt.close(fig)

        # ── 3. 权重分布直方图 ──
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(w, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
        ax.axvline(w.mean(), color='red', linestyle='--', label=f'mean={w.mean():.3f}')
        ax.set_xlabel("Weight")
        ax.set_ylabel("Count")
        ax.set_title("Weight Distribution")
        ax.legend()
        plt.tight_layout()
        res["weight_histogram"] = wandb.Image(fig)
        plt.close(fig)

        # ── 4. Top-K 高权重点 + flow 向量 ──
        K = min(50, pts.shape[0])
        topk_idx = np.argsort(w)[-K:]
        # 高权重点 (绿色) vs 低权重点 (灰色)
        topk_pts = pts[topk_idx]
        bottomk_idx = np.argsort(w)[:K]
        bottomk_pts = pts[bottomk_idx]
        # # 高权重点 + flow 末端
        # flow_end = topk_pts + pred_f[topk_idx]
        # red = np.tile([255, 0, 0], (K, 1))
        # 拼接: 低权灰 + 高权绿(起点) + 高权红(flow末端)
        gray = np.zeros((K, 3)) + 128
        green = np.tile([0, 255, 0], (K, 1))
        all_pts = np.concatenate([
            np.concatenate([bottomk_pts, gray], axis=1),
            np.concatenate([topk_pts, green], axis=1),
        ], axis=0)
        res["topk_weighted_pts"] = wandb.Object3D(all_pts)

        model.train()
        return res