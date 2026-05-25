import copy
import torch
import wandb
from pytorch3d.pytorch3d.transforms.transform3d import Transform3d
from taxpose.training.point_cloud_training_module import PointCloudTrainingModule
from taxpose.nets.RL_policy import PolicyModel
from taxpose.utils.se3 import (
    dense_flow_loss,
    dualflow2pose,
    flow2pose,
    get_degree_angle,
    get_translation,
    mse_criterion,
)


class RLTrainingModule(PointCloudTrainingModule):
    def __init__(
        self,
        model: PolicyModel,
        lr=1e-3,
        lr_cfg: dict = {
            "scheduler": "constant",
            "max_steps": 400,
            "warmup_ratio": 0.1,
            "min_lr": 1e-5,
            "by_epoch": True,
        },
        image_log_period=500,
        action_weight=1,
        anchor_weight=1,
        flow_supervision="both",  # ('both', 'action2anchor', 'anchor2action')
        optimization_mode: str = "auto",
        kl_coef: float = 0.05,      # KL 惩罚系数
        clip_eps: float = 0.2,      # PPO-style 裁剪阈值
        update_base_every: int = 5,  # base model 更新频率
        tensorboard_writer=None,
    ):
        super().__init__(
            model=model,
            lr=lr,
            image_log_period=image_log_period,
            tensorboard_writer=tensorboard_writer,
            optimization_mode=optimization_mode,
            **lr_cfg,
        )
        self.model: PolicyModel = model
        self.lr = lr
        self.image_log_period = image_log_period
        self.action_weight = action_weight
        self.anchor_weight = anchor_weight
        self.flow_supervision = flow_supervision
        # GRPO 专用超参数
        self.kl_coef = kl_coef
        self.clip_eps = clip_eps
        self.update_base_every = update_base_every
        # 创建并冻结参考模型（base model）
        self.base_model = copy.deepcopy(model)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

    def on_train_epoch_start(self) -> None:
        if self.current_epoch > 0 and \
                self.current_epoch % self.update_base_every == 0:
            self.base_model.load_state_dict(self.model.state_dict())
        return super().on_train_epoch_start()

    def module_step(self, batch, batch_idx):
        points_trans_action = batch["points_action_trans"]  # B,N,3
        points_trans_anchor = batch["points_anchor_trans"]

        bz, n, _ = points_trans_action.shape
        if self.training:
            with torch.no_grad():
                samples = self.model.rl_sample(points_trans_action, points_trans_anchor)

            log_values = {}
            loss, log_values = self.compute_loss(
                samples, batch, log_values=log_values, loss_prefix=""
            )
        else:
            outputs = self.model.forward(points_trans_action, points_trans_anchor)
            loss, log_values = self.compute_error(outputs, batch, log_values)
        return loss, log_values

    def compute_loss(self, samples, batch, log_values={}, loss_prefix=""):
        """
        samples: 包含
            - state_act, state_anch: 状态（点云）
            - flow_act, flow_anch: 当前策略采样的动作
            - adv: 预先计算好的组内标准化优势 (batch_size,)
        batch:  原始 batch（可能含有 reward，但这里优势已算好，无需使用）
        """
        state_act = batch["points_action_trans"]  # B,N,3
        state_anch = batch["points_anchor_trans"]
        flow_act = samples['flow_act']  # B, 3, N ??
        flow_anch = samples['flow_anch']
        adv = samples['adv']                     # 形状 [B, G]，组内标准化后的优势

        # ---- 当前策略的对数概率 ----
        log_p = self.model.log_probs(state_act, state_anch, flow_act, flow_anch)  # [B, G]
        # ---- 参考策略的对数概率（冻结，无梯度） ----
        with torch.no_grad():
            self.base_model.eval()
            log_p_base = self.base_model.log_probs(state_act, state_anch, flow_act, flow_anch)
        log_p_base = log_p_base.detach()          # [B, G]

        # ---- GRPO 策略损失 (带裁剪) ----
        log_ratio = log_p - log_p_base
        # 安全范围，避免 exp 溢出
        log_ratio_clamped = torch.clamp(log_ratio, min=-20.0, max=20.0)
        # 稳定的 ratio 和 clipped ratio
        ratio = torch.exp(log_ratio_clamped)   # 重要性采样比率
        surr1 = adv * ratio
        surr2 = adv * torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        grpo_loss = -torch.mean(torch.min(surr1, surr2))

        # ---- KL 散度（reverse KL: E[log p_base - log p]） ----
        kl = (log_p_base - log_p).mean()          # 近似 KL(policy || base)
        # 也可用 forward KL: (log_p - log_p_base).mean()，视需要选择

        # ---- 总损失 ----
        loss = grpo_loss + self.kl_coef * kl
        # ---- 日志记录 ----
        log_values[loss_prefix + "kl"] = kl.detach()
        log_values[loss_prefix + "grpo_loss"] = grpo_loss.detach()
        log_values[loss_prefix + "loss"] = loss.detach()
        # 可选：记录比率范围和优势均值
        log_values[loss_prefix + "ratio_mean"] = ratio.detach().mean()
        log_values[loss_prefix + "ratio_max"] = ratio.detach().max()
        log_values[loss_prefix + "adv_mean"] = adv.mean()

        return loss, log_values
    
    def compute_error(self, model_output, batch, log_values):
        x_action = model_output["flow_action"]
        x_anchor = model_output["flow_anchor"]

        # The original point clouds from the demonstration.
        points_action = batch["points_action"]  # action point clouds
        points_anchor = batch["points_anchor"]  # anchor point clouds
        # The point clouds transformed by the ground truth transforms.
        points_trans_action = batch["points_action_trans"]  # T0 -> action point clouds
        points_trans_anchor = batch["points_anchor_trans"]  # T1 -> anchor point clouds
        input_act_pts = points_trans_action
        inputs_anch_pts = points_trans_anchor

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
            inputs_anch_pts = model_output["anch_down_sample"]

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
                training=True  # self.training,
            )

            error_R_max, error_R_min, error_R_mean = get_degree_angle(
                T0.inverse().compose(T1).compose(pred_T_action.inverse())
            )
            error_t_max, error_t_min, error_t_mean = get_translation(
                T0.inverse().compose(T1).compose(pred_T_action.inverse())
            )

            # Loss associated with ground truth transform
            pred_points_action = pred_T_action.transform_points(points_trans_action)
            points_action_target = T1.transform_points(points_action)
            point_loss_action = mse_criterion(pred_points_action, points_action_target)

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
            smoothness_loss_action = mse_criterion(
                pred_flow_action, induced_flow_action
            )
            # loss associated with dense flow
            # pred_T_action=T1T0^-1
            gt_T_action = T0.inverse().compose(T1)
            dense_loss_action = dense_flow_loss(
                points=input_act_pts,
                flow_pred=pred_flow_action,
                trans_gt=gt_T_action,
            )

            pred_T_anchor = pred_T_action.inverse()
            # Loss associated with ground truth transform
            pred_points_anchor = pred_T_anchor.transform_points(points_trans_anchor)
            points_anchor_target = T0.transform_points(points_anchor)
            point_loss_anchor = mse_criterion(
                pred_points_anchor,
                points_anchor_target,
            )

            # Loss associated flow vectors matching a consistent rigid transform
            induced_flow_anchor = (
                pred_T_anchor.transform_points(inputs_anch_pts) - inputs_anch_pts
            ).detach()
            smoothness_loss_anchor = mse_criterion(
                pred_flow_anchor,
                induced_flow_anchor,
            )

            # loss associated with dense flow
            # pred_T_action=T1T0^-1
            gt_T_anchor = T1.inverse().compose(T0)
            dense_loss_anchor = dense_flow_loss(
                points=inputs_anch_pts,
                flow_pred=pred_flow_anchor,
                trans_gt=gt_T_anchor,
            )

            self.action_weight = (self.action_weight) / (
                self.action_weight + self.anchor_weight
            )
            self.anchor_weight = (self.anchor_weight) / (
                self.action_weight + self.anchor_weight
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
            point_loss_action = mse_criterion(
                pred_points_action,
                points_action_target,
            )

            # Loss associated flow vectors matching a consistent rigid transform
            smoothness_loss_action = mse_criterion(
                pred_flow_action,
                induced_flow_action,
            )

            dense_loss_action = dense_flow_loss(
                points=points_trans_action,
                flow_pred=pred_flow_action,
                trans_gt=gt_T_action,
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
            point_loss_anchor = mse_criterion(
                pred_points_anchor,
                points_anchor_target,
            )

            # Loss associated flow vectors matching a consistent rigid transform
            smoothness_loss_anchor = mse_criterion(
                pred_flow_anchor,
                induced_flow_anchor,
            )
            dense_loss_anchor = dense_flow_loss(
                points=points_trans_anchor,
                flow_pred=pred_flow_anchor,
                trans_gt=gt_T_anchor,
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

        loss = (
            self.displace_loss_weight * point_loss,
            self.consistency_loss_weight * smoothness_loss,
            self.direct_correspondence_loss_weight * dense_loss,
        )

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

                anch_perm = torch.randperm(inputs_anch_pts.shape[1])
                anchor_points = inputs_anch_pts[:, anch_perm, :]

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
            loss_indirect = ww * mse_criterion(
                pred_flow_action, shuffled_act_corr
            ) + ww * mse_criterion(pred_flow_anchor, shuffled_anch_corr)
            loss += (loss_indirect,)
            log_values[loss_prefix + "indirect_loss"] = loss_indirect.detach()
            del shuffled_output, shuffled_act_corr, shuffled_anch_corr
            del act_perm, anch_perm, inv_perm_a, inv_perm_b
            torch.cuda.empty_cache()  # 清空缓存

        if (
            self.res_smooth_loss_weight > 0.0
            and self.current_epoch >= self.res_smooth_start_epoch
        ):
            act_res_smooth_loss = self.compute_res_smoothness_loss(
                model_output["residual_flow_action"], input_act_pts
            )
            anch_res_smooth_loss = self.compute_res_smoothness_loss(
                model_output["residual_flow_anchor"], inputs_anch_pts
            )
            loss_res = (
                self.res_smooth_loss_weight * act_res_smooth_loss
                + self.res_smooth_loss_weight * anch_res_smooth_loss
            )
            loss += (loss_res,)
            log_values[loss_prefix + "res_smooth_loss"] = loss_res.detach()

        if "act_emb_similarity" in model_output:
            loss += (self.compute_emb_sim_loss(model_output, log_values, loss_prefix),)

        log_values[loss_prefix + "point_loss"] = self.displace_loss_weight * point_loss
        log_values[loss_prefix + "smoothness_loss"] = (
            self.consistency_loss_weight * smoothness_loss
        )
        log_values[loss_prefix + "dense_loss"] = (
            self.direct_correspondence_loss_weight * dense_loss
        )
        centered_pred_ps_A = pred_points_action.detach()
        centered_pred_ps_A = centered_pred_ps_A - centered_pred_ps_A.mean(
            dim=1, keepdim=True
        )
        centered_gt_ps = points_action_target.detach()
        centered_gt_ps = centered_gt_ps - centered_gt_ps.mean(dim=1, keepdim=True)
        # log_values[loss_prefix + "only_Rotate_L2_pcs_distance"] = \
        #     mse_criterion(centered_pred_ps_A, centered_gt_ps)
        log_values[loss_prefix + "R0_mean"] = R0_mean
        log_values[loss_prefix + "R0_max"] = R0_max
        log_values[loss_prefix + "R0_min"] = R0_min
        log_values[loss_prefix + "R1_mean"] = R1_mean
        log_values[loss_prefix + "R1_max"] = R1_max
        log_values[loss_prefix + "R1_min"] = R1_min

        log_values[loss_prefix + "t0_mean"] = t0_mean
        log_values[loss_prefix + "t0_max"] = t0_max
        log_values[loss_prefix + "t0_min"] = t0_min
        log_values[loss_prefix + "t1_mean"] = t1_mean
        log_values[loss_prefix + "t1_max"] = t1_max
        log_values[loss_prefix + "t1_min"] = t1_min

        log_values[loss_prefix + "error_R_mean"] = error_R_mean
        log_values[loss_prefix + "error_t_mean"] = error_t_mean

        return loss, log_values

