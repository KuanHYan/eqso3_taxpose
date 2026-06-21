import copy
from typing import Any, Mapping
import wandb
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pytorch3d.transforms import Transform3d
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
from taxpose.utils.lr import MilestoneScheduler, LinearAnnealingWarmup


class RLTrainingModule(PointCloudTrainingModule):
    def __init__(
        self,
        model,
        lr=1e-6,
        lr_cfg: dict = {
            "scheduler": "constant",
            "max_steps": 400,
            "warmup_ratio": 0.1,
            "min_lr": 1e-7,
            "by_epoch": True,
            "weight_decay": 0.0,
        },
        image_log_period=500,
        action_weight=1,
        anchor_weight=1,
        flow_supervision="both",  # ('both', 'action2anchor', 'anchor2action')
        optimization_mode="auto",
        kl_coef: float = 0.05,      # KL 惩罚系数
        clip_eps: float = 0.2,      # PPO-style 裁剪阈值
        grpo_iter: int = 10,        # GRPO单词迭代次数
        update_base_every: int = 1000,  # base model 更新频率
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
        self.model = model
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
        self.grpo_iter = grpo_iter

    def on_train_epoch_start(self) -> None:
        if self.current_epoch % self.update_base_every == 0:
            self.print(f"update base model at: {self.current_epoch}")
            self.base_model.load_state_dict(self.model.state_dict())
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        self.train()
        opt = self.optimizers()        # 获取优化器
        sch = self.lr_schedulers()     # 获取调度器
        log_values = {}
        # 采样（无梯度）
        # 预先计算 log_p_old 与 log_p_base（均为 eval 或 no_grad）
        T0 = Transform3d(matrix=batch["T0"])
        T1 = Transform3d(matrix=batch["T1"])
        gt_trans = T0.inverse().compose(T1)
        with torch.no_grad():
            samples = self.model.rl_sample(
                batch["points_action_trans"],
                batch["points_anchor_trans"],
                return_cache=True,
                gt_trans=gt_trans
            )
            state_act = batch["points_action_trans"]  # B,N,3
            state_anch = batch["points_anchor_trans"]
            act_1 = samples['act_1']  # B, G, N, 3
            act_2 = samples['act_2']
            samples["log_p_old"] = samples["log_prob"]
            samples["log_p_base"] = self.base_model.log_probs(
                state_act, state_anch, act_1, act_2, cache=samples["cache"])

        if self._automatic_optimization:
            total_grpo_loss = self.compute_loss(samples, batch, log_values)
        else:
            total_grpo_loss = 0.0
            for _ in range(self.grpo_iter):
                opt.zero_grad()
                loss = self.compute_loss(samples, batch, log_values)
                loss.backward()
                self.clip_gradients(opt, 1.0, 'norm')
                opt.step()
                total_grpo_loss += loss.detach()

            # 每个 batch 结束才步进调度器
            if sch is not None:
                sch.step()

        for key, val in log_values.items():
            self.log(key, val, logger=True, sync_dist=True)

        if (self.global_step % self.image_log_period) == 0 and \
                self.trainer.is_global_zero:
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if isinstance(val, wandb.Object3D) and wandb.run is not None:
                    wandb.log(
                        {
                            key: val,
                            "trainer/global_step": self.global_step,
                        }
                    )
                elif self.logger is not None:
                    self.logger.log_image(
                        key,
                        images=[val],  # self.global_step
                    )
        self.log("mean rewrad", log_values['reward_mean'], prog_bar=True, logger=True, sync_dist=True)
        # avg_loss = total_grpo_loss / self.grpo_iter
        # self.log("train_loss", avg_loss)
        return total_grpo_loss

    def module_step(self, batch, batch_idx):
        points_trans_action = batch["points_action_trans"]  # B,N,3
        points_trans_anchor = batch["points_anchor_trans"]
        log_values = {}
        outputs = self.model(points_trans_action, points_trans_anchor)
        loss, log_values = self.compute_error(outputs, batch, log_values)
        return loss, log_values

    def compute_loss(self, samples, batch, log_values={}, loss_prefix=''):
        """
        samples: 包含
            - state_act, state_anch: 状态（点云）
            - flow_act, flow_anch: 当前策略采样的动作
            - adv: 预先计算好的组内标准化优势 (batch_size,)
        batch:  原始 batch（可能含有 reward，但这里优势已算好，无需使用）
        """
        state_act = batch["points_action_trans"]  # B,N,3
        state_anch = batch["points_anchor_trans"]
        act_1 = samples['act_1']  # B, 3, N ??
        act_2 = samples['act_2']
        adv = samples['adv']            # 形状 [B, G]，组内标准化后的优势
        log_p_base = samples['log_p_base']
        log_p_old = samples['log_p_old']

        # ---- 当前策略的对数概率 ----
        log_p = self.model.log_probs(
            state_act, state_anch, act_1, act_2, samples['cache'])  # [G, B]
        # ---- GRPO 策略损失 (带裁剪) ----
        log_ratio = log_p - log_p_old
        # 安全范围，避免 exp 溢出
        # log_ratio_clamped = torch.clamp(log_ratio, min=-10, max=10)
        # 稳定的 ratio 和 clipped ratio
        ratio = torch.exp(log_ratio)   # 重要性采样比率
        surr1 = adv * ratio
        surr2 = adv * torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        grpo_loss = -torch.mean(torch.min(surr1, surr2))

        # ---- KL 散度（reverse KL: E[log p_base - log p]） ----
        ref_logp_delat = log_p_base - log_p
        kl = (torch.exp(ref_logp_delat) - ref_logp_delat - 1).mean()  #  近似 KL(policy || base)
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
        log_values[loss_prefix + "adv_std"] = samples['reward_std'].mean()
        log_values["reward_mean"] = samples['reward_mean'].mean()

        return loss

    def compute_error(self, model_output, batch, log_values, loss_prefix=""):
        x_action = model_output["flow_action"]
        x_anchor = model_output.get("flow_anchor", None)

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

        # Extract predictred flow and weight
        pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)

        if points_trans_action.shape[1] != pred_flow_action.shape[1]:
            input_act_pts = model_output["act_down_sample"]
            inputs_anch_pts = model_output.get("anch_down_sample", None)

        if x_anchor is None:
            pred_T_action = flow2pose(
                xyz=input_act_pts,
                flow=pred_flow_action,
                weights=pred_w_action,
                return_transform3d=True,
            )        
        else:
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)
            pred_T_action = dualflow2pose(
                xyz_src=input_act_pts,
                xyz_tgt=inputs_anch_pts,
                flow_src=pred_flow_action,
                flow_tgt=pred_flow_anchor,
                weights_src=pred_w_action,
                weights_tgt=pred_w_anchor,
                return_transform3d=True,
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

        if x_anchor is None:
            point_loss_anchor = point_loss_action.detach()
            dense_loss_anchor = dense_loss_action.detach()
            smoothness_loss_anchor = smoothness_loss_action.detach()
        else:
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
            point_loss,
            smoothness_loss,
            dense_loss,
        )
        log_values[loss_prefix + "point_loss"] = point_loss
        log_values[loss_prefix + "smoothness_loss"] = smoothness_loss
        log_values[loss_prefix + "dense_loss"] = dense_loss
        centered_pred_ps_A = pred_points_action.detach()
        centered_pred_ps_A = centered_pred_ps_A - centered_pred_ps_A.mean(
            dim=1, keepdim=True
        )
        centered_gt_ps = points_action_target.detach()
        centered_gt_ps = centered_gt_ps - centered_gt_ps.mean(dim=1, keepdim=True)

        log_values[loss_prefix + "error_R_mean"] = error_R_mean
        log_values[loss_prefix + "error_t_mean"] = error_t_mean

        return loss, log_values

    # ─── Weight 可视化 (人工分析逐点权重质量) ───

    @torch.no_grad()
    def _make_weight_visualizations(self, batch, batch_idx,
                                     max_samples: int = 4):
        """生成逐点权重的多种可视化, 返回 wandb 日志字典.

        可视化方案:
          1. weight_colored_pts_b{N}  — 点云按权重着色 (红=高权, 蓝=低权)
          2. weight_vs_error_b{N}     — 权重 vs flow 误差散点图 (检验相关性)
          3. weight_histogram_b{N}    — 权重分布直方图
          4. topk_weighted_pts_b{N}  — 高亮 Top-K 权重点的 flow 向量
        """
        self.model.eval()
        pts_trans_action = batch["points_action_trans"]   # (B, N_in, 3)
        pts_trans_anchor = batch["points_anchor_trans"]
        T0 = Transform3d(matrix=batch["T0"])
        T1 = Transform3d(matrix=batch["T1"])
        gt_T_action = T0.inverse().compose(T1)

        B = min(pts_trans_action.shape[0], max_samples)
        res = {}

        b = torch.randint(0, B, (1,))
        try:
            # ── 确定性前向, 拿到逐点 flow 与 weight ──
            model_output = self.model(
                pts_trans_action[b:b+1], pts_trans_anchor[b:b+1])
            flow_act = model_output["flow_action"]           # (1, N_out, C)

            # flow_act 通道: [:,:,:3] = flow, [:,:,3] = weight logit（如果 pred_weight）
            if flow_act.shape[-1] < 4:
                # pred_weight=False 时没有 weight 通道, 跳过该样本
                return res

            pred_flow = flow_act[0, :, :3]                  # (N_out, 3)
            pred_w = torch.sigmoid(flow_act[0, :, 3])       # (N_out,)
            N_out = pred_flow.shape[0]

            # GT 刚性流: 对模型输出对应的点集计算 GT flow
            # 注意: Coarse_Res_Head 升采样时 N_out ≠ N_in,
            #  此时用模型输出的点数作为基准, 从输入中 FPS 采样匹配
            if N_out == pts_trans_action.shape[1]:
                pts_for_gt = pts_trans_action[b:b+1]         # (1, N, 3)
            else:
                # 点数不匹配 (升采样/降采样场景):
                # 对输入点做随机采样或 FPS 以对齐 N_out
                n_in = pts_trans_action.shape[1]
                if N_out < n_in:
                    idx = torch.randperm(n_in, device=pts_trans_action.device)[:N_out]
                    pts_for_gt = pts_trans_action[b:b+1, idx, :]
                else:
                    # N_out > n_in: 重复采样
                    idx = torch.randint(0, n_in, (N_out,), device=pts_trans_action.device)
                    pts_for_gt = pts_trans_action[b:b+1, idx, :]

            gt_flow = (gt_T_action.transform_points(pts_for_gt) - pts_for_gt)[0]  # (N_out, 3)

            # 逐点 flow 误差
            flow_err = (pred_flow - gt_flow).norm(p=2, dim=-1)  # (N_out,)

            pts = pts_for_gt[0].cpu().numpy()                   # (N_out, 3)
            w = pred_w.cpu().numpy()
            ferr = flow_err.cpu().numpy()

            # ── 1. 按权重着色的点云 ──
            w_min, w_max = w.min(), w.max()
            w_range = w_max - w_min
            w_norm = (w - w_min) / (w_range + 1e-8) if w_range > 1e-8 else np.full_like(w, 0.5)
            colors = np.stack([w_norm, 0.2, 1 - w_norm], axis=-1) * 255  # 红→蓝
            pts_rgb = np.concatenate([pts, colors], axis=1)
            res[f"weight_colored_pts_b{b}"] = wandb.Object3D(pts_rgb)

            # ── 2. 权重 vs flow 误差散点图 ──
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # 散点
            axes[0].scatter(w, ferr, c=w_norm, cmap='coolwarm',
                            alpha=0.5, s=10, edgecolors='none')
            axes[0].set_xlabel("Predicted Weight")
            axes[0].set_ylabel("Flow Error (L2)")
            axes[0].set_title(f"Sample {b}: Weight vs Flow Error")
            # 分箱统计 (binned mean ± std)
            bins = np.linspace(0, 1, 11)
            bin_idx = np.clip(np.digitize(w, bins) - 1, 0, 9)
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
            res[f"weight_vs_error_b{b}"] = wandb.Image(fig)
            plt.close(fig)

            # ── 3. 权重分布直方图 ──
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(w, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
            ax.axvline(w.mean(), color='red', linestyle='--',
                        label=f'mean={w.mean():.3f}')
            ax.set_xlabel("Weight")
            ax.set_ylabel("Count")
            ax.set_title(f"Sample {b}: Weight Distribution")
            ax.legend()
            plt.tight_layout()
            res[f"weight_histogram_b{b}"] = wandb.Image(fig)
            plt.close(fig)

            # ── 4. Top-K 高权重点 + flow 向量 ──
            K = min(50, pts.shape[0])
            topk_idx = np.argsort(w)[-K:]
            bottomk_idx = np.argsort(w)[:K]
            topk_pts = pts[topk_idx]
            bottomk_pts = pts[bottomk_idx]

            gray = np.full((K, 3), 128)
            green = np.tile([0, 255, 0], (K, 1))
            all_pts = np.concatenate([
                np.concatenate([bottomk_pts, gray], axis=1),
                np.concatenate([topk_pts, green], axis=1),
            ], axis=0)
            res[f"topk_weighted_pts_b{b}"] = wandb.Object3D(all_pts)

        except Exception as e:
            self.print(f"[WARN] weight viz sample {b} failed: {e}")
        finally:
            del model_output
            torch.cuda.empty_cache()

        self.model.train()
        return res

    # def visualize_results(self, batch, batch_idx):
    #     """重载: 追加权重可视化."""
    #     res = super().visualize_results(batch, batch_idx)
    #     try:
    #         weight_viz = self._make_weight_visualizations(batch, batch_idx)
    #         res.update(weight_viz)
    #     except Exception as e:
    #         # 可视化失败不应中断训练
    #         self.print(f"[WARN] weight visualization failed: {e}")
    #     return res

    def configure_optimizers(self):
        # 使用 Policy 的 get_param_groups 接口, 支持不同参数组使用不同学习率
        # SE3PolicyModel 的方差头 (translate_var / rotate_var) 会获得更高 LR
        if hasattr(self.model, 'get_param_groups'):
            param_groups = self.model.get_param_groups(self.lr)
        else:
            param_groups = self.parameters()

        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.weight_decay)
        self._optimizer_ = optimizer

        if self.warmup_steps <= 0:
            return optimizer

        if self.lr_scheduler == 'constant':
            milestones = [self.end_lr_steps]
            self.scheduler = scheduler = MilestoneScheduler(
                optimizer,
                milestones=milestones, gamma=1.0,
                max_lr=self.lr, min_lr=self.min_lr,
                warmup_steps=self.warmup_steps,
            )
        elif self.lr_scheduler == 'milestone':
            milestones = [int(self.end_lr_steps * stone)
                          for stone in [0.5, 0.75, 0.9]]
            self.scheduler = scheduler = MilestoneScheduler(
                optimizer,
                milestones=milestones, gamma=0.5,
                max_lr=self.lr, min_lr=self.min_lr,
                warmup_steps=self.warmup_steps,
            )
        elif self.lr_scheduler == 'linear':
            self.scheduler = scheduler = LinearAnnealingWarmup(
                optimizer,
                total_steps=self.end_lr_steps,
                max_lr=self.lr, min_lr=self.min_lr,
                warmup_steps=self.warmup_steps,
            )
        else:
            return optimizer

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch' if self.by_epoch else 'step',
            },
        }

    def extract_flow_and_weight(self, x):
        # x: Batch, num_points, 4
        pred_flow = x[:, :, :3]
        if x.shape[2] > 3:
            pred_w = torch.sigmoid(x[:, :, 3])
        else:
            pred_w = None
        return pred_flow, pred_w


if __name__ == "__main__":
    from taxpose.nets.raw_dgcnn import DGCNNArgs
    from taxpose.nets.head import HeadConfig

    torch.cuda.set_device(1)

    encoder_args = DGCNNArgs(
        name="raw_dgcnn",
        emb_dims=512,
        knn=2,
    )
    head = HeadConfig(
        norm=torch.nn.LayerNorm,
        head_type="rl_residual",
        emb_dims=512,
        project_corrs=True,
        project_corrs_mode='vn',
        output_num=1024,
    )
    
    net = PolicyModel(
        encoder_args,
        head,
        "/home/yan/pose_estimation/taxpose/logs/rl_reward/2026-05-24/10-51-49/checkpoints/last.ckpt",
    )
    model = RLTrainingModule(net)
    model.load_state_dict(torch.load('/home/yan/pose_estimation/taxpose/logs/train_taxpose/best_ckpt/vn_Wab_wo_TFhead_6400dz.ckpt')['state_dict'])