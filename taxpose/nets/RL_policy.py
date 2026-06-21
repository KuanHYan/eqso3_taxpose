import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, quaternion_multiply
from taxpose.nets.transformer_flow import ResidualFlow_DiffEmbTransformer
from taxpose.nets.transformer_flow_pm import CustomTransformer
from taxpose.nets.RL_tune import RewardModel
from taxpose.utils.se3 import dualflow2pose, flow2pose, get_degree_angle, get_translation


class PolicyModel(ResidualFlow_DiffEmbTransformer):
    def __init__(
        self,
        encoder_cfg,
        head_cfg,
        reward_model_path=None,
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
        group=32,
        n_blocks=1,
        attn_mode="torch_attn",
        manual_reawrd=False,
    ):
        super(PolicyModel, self).__init__(
            encoder_cfg,
            head_cfg,
            cycle,
            center_feature,
            freeze_embnn,
            return_attn,
            multilaterate,
            mlat_sample,
            mlat_nkps,
            feature_channels,  # Number of extra channels we'll pass into the network.
            conditional,
            dropout,
            pos_encoding,
            n_blocks,
            attn_mode=attn_mode
        )
        assert reward_model_path is not None or manual_reawrd == False
        if not manual_reawrd:
            self.reward_model = RewardModel(encoder_cfg, cycle, False, feature_channels, dropout, False)
            self.reward_load_state_dict(reward_model_path)
            self.reward_model.eval()
        self.manual_reawrd = manual_reawrd
        self.group = group

    def get_param_groups(self, base_lr: float, **kwargs):
        """返回优化器参数组 (PolicyModel).

        PolicyModel 没有额外的方差网络, 所有参数使用相同的 base_lr.
        """
        return [{"params": self.parameters(), "lr": base_lr}]

    def reward_load_state_dict(self, path):
        state_dict = torch.load(path)['state_dict']
        # remove 'model.'
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict[k[len('model.'):]] = state_dict.pop(k)
        self.reward_model.load_state_dict(state_dict)
        self.reward_model.requires_grad_(False)

    def ideal_reward(self,
                     pred_flow_action, pred_w_action,       # 原始采样 flow (未经过刚体约束)
                     pred_trans: Transform3d, s_next,       # 由 flow 求解的刚体变换及结果
                     act_pts, anchor_pts,                   # 点云
                     gt_trans: Transform3d,                 # GT 变换
                     pred_flow_anchor=None, pred_w_anchor=None):
        """计算与 compute_error 评估指标对齐的多目标 Reward。

        关键: 使用原始采样 flow (pred_flow_action) 而非 s_next 来计算 smoothness 和 dense，
        这样才能真正反映 flow 的自洽性和与 GT 刚性流的偏差。

        对应 compute_error 中的指标:
          - point_loss      → r_point   (变换后点云与 GT 的 L2 距离)
          - error_R_mean    → r_pose    (旋转角度误差)
          - error_t_mean    → r_pose    (平移误差)
          - smoothness_loss → r_smooth  (原始 flow 与刚体诱导流的偏差，反映 flow 自洽性)
          - dense_loss      → r_dense   (原始 flow 与 GT 刚性流的偏差)

        所有 reward 分量均通过 1/(x+eps) 转换为正值，clamp 后加权求和。
        """
        # ---- 将 GT 变换广播到 group 维度 ----
        gt_trans_mat = gt_trans.get_matrix().unsqueeze(0).expand(self.group, -1, -1, -1).reshape(-1, 4, 4)
        gt_trans_group = Transform3d(matrix=gt_trans_mat)

        # ---- 1. point_loss 对应: 变换后点云与 GT 变换后点云的 L2 距离 ----
        gt_transformed_pts = gt_trans_group.transform_points(act_pts)
        point_dist = (s_next - gt_transformed_pts).norm(p=2, dim=-1).mean(dim=-1)
        r_point = torch.clamp_max(1.0 / (point_dist + 1e-6), max=1e6)

        # # ---- 2. error_R + error_t 对应: 位姿误差 ----
        # error_mat = gt_trans_group.compose(pred_trans.inverse())
        # error_R = get_degree_angle(error_mat, return_batch=True)[0]        # 旋转误差 (度)
        # error_t = get_translation(error_mat, return_batch=True)[0]         # 平移误差
        # r_pose = torch.clamp_max(1.0 / (error_R + error_t + 1e-6), max=1e6)

        # ---- 3. smoothness_loss 对应: 原始采样 flow vs 刚体诱导流 ----
        # compute_error 中: smoothness_loss = mse(pred_flow, induced_flow)
        # 其中 induced_flow = pred_T(p) - p，即刚体变换严格定义的流
        # 原始采样 flow 与刚体流的偏差越大 → flow 越不自洽 → 惩罚
        induced_flow_rigid = s_next - act_pts               # 刚体诱导流 (已经是变换后-变换前)
        smoothness_dist = (pred_flow_action - induced_flow_rigid).norm(p=2, dim=-1).mean(dim=-1)
        r_smooth = torch.clamp_max(1.0 / (smoothness_dist + 1e-6), max=1e6)

        # ---- 4. dense_loss 对应: 原始采样 flow vs GT 刚性流 ----
        # compute_error 中: dense_loss = dense_flow_loss(points, flow_pred, trans_gt)
        # 即 pred_flow 与 gt_T(p) - p 的偏差
        gt_induced_flow = gt_transformed_pts - act_pts      # GT 刚性流
        dense_dist = (pred_flow_action - gt_induced_flow).norm(p=2, dim=-1).mean(dim=-1)
        r_dense = torch.clamp_max(1.0 / (dense_dist + 1e-6), max=1e6)

        # ---- 5. (可选) anchor 侧的 dense_loss ----
        if pred_flow_anchor is not None:
            # GT 刚性流从 anchor 侧: gt_T_anchor(p) - p = gt_T^{-1}(p) - p
            gt_induced_flow_anch = gt_trans_group.inverse().transform_points(anchor_pts) - anchor_pts
            dense_dist_anch = (pred_flow_anchor - gt_induced_flow_anch).norm(p=2, dim=-1).mean(dim=-1)
            r_dense_anch = torch.clamp_max(1.0 / (dense_dist_anch + 1e-6), max=1e6)
            r_dense = (r_dense + r_dense_anch) / 2.0

        # ---- 加权组合 (权重可调) ----
        # point 和 pose 是核心指标，smooth 和 dense 作为辅助正则
        reward = r_point + 0.1 * r_smooth + 0.5 * r_dense
        return reward

    @torch.no_grad()
    def compute_reward(
            self, act_pts, anchor_pts,
            pred_flow_action, pred_w_action,
            pred_flow_anchor=None, pred_w_anchor=None,
            gt_trans=None
    ):
        bz, _, n = act_pts.shape
        act_pts = act_pts.permute(0, 2, 1).contiguous().unsqueeze(0).expand(self.group, -1, -1, -1).reshape(-1, n, 3)
        anchor_pts = anchor_pts.permute(0, 2, 1).contiguous().unsqueeze(0).expand(self.group, -1, -1, -1).reshape(-1, n, 3)
        pred_flow_action = pred_flow_action.reshape(-1, n, 3)
        pred_w_action = pred_w_action.reshape(-1, n)
        if pred_flow_anchor is None:
            pred_trans = flow2pose(
                act_pts, pred_flow_action,
                weights=pred_w_action,
                return_transform3d=True,
            )
        else:
            pred_flow_anchor = pred_flow_anchor.reshape(-1, n, 3)
            pred_w_anchor = pred_w_anchor.reshape(-1, n)
            pred_trans = dualflow2pose(
                xyz_src=act_pts,
                xyz_tgt=anchor_pts,
                flow_src=pred_flow_action,
                flow_tgt=pred_flow_anchor,
                weights_src=pred_w_action,
                weights_tgt=pred_w_anchor,
                return_transform3d=True,
                training=True,
            )
        s_next = pred_trans.transform_points(act_pts)
        # ## Debug ##############
        # gt_trans_mat = gt_trans.get_matrix().unsqueeze(0).expand(self.group, -1, -1, -1).reshape(-1, 4, 4)
        # gt_trans_group = Transform3d(matrix=gt_trans_mat)
        # matrix_dot = gt_trans_group.compose(real_act.inverse())
        # error_R_mean, _, _ = get_degree_angle(matrix_dot, return_batch=True)
        # error_t_mean, _, _ = get_translation(matrix_dot, return_batch=True)
        # #######################
        assert s_next.shape == anchor_pts.shape
        if self.manual_reawrd:
            assert gt_trans is not None
            rewards = self.ideal_reward(
                pred_flow_action, pred_w_action,
                pred_trans, s_next,
                act_pts, anchor_pts, gt_trans,
                pred_flow_anchor, pred_w_anchor)
        else:
            rewards = self.reward_model(s_next, anchor_pts, return_total_reward=True)
        return rewards, pred_trans

    @torch.no_grad()
    def rl_sample(self, *input, return_cache=False, gt_trans=None):
        (
            action_points, anchor_points,
            action_embedding, anchor_embedding,
            action_pt_pos, anchor_pt_pos,
            act_down_sample, anch_down_sample
        ) = self._embedding(*input)

        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn
        ) = self._backbone(action_embedding, anchor_embedding, action_pt_pos, anchor_pt_pos)

        head_action_output = self.head_action.sample(
            action_embedding_tf,
            action_embedding,
            anchor_embedding,
            action_points,
            anchor_points,
            act_down_sample,
            anch_down_sample,
            scores=action_attn,
            sample_num=self.group,
            return_logP=True,
        )
        flow_act = head_action_output['samples']

        weight_act = torch.sigmoid(head_action_output['weights'])
        assert weight_act.shape == (self.group, flow_act.shape[1], flow_act.shape[2]), "weight shape mismatch"
        logP = head_action_output.get('log_probs', None)  # G, B
        del head_action_output

        if self.cycle:
            head_anchor_output = self.head_anchor.sample(
                anchor_embedding_tf,
                anchor_embedding,
                action_embedding,
                anchor_points,
                action_points,
                anch_down_sample,
                act_down_sample,
                scores=anchor_attn,
                sample_num=self.group,
                return_logP=True,
            )
            flow_anch = head_anchor_output['samples']
            weight_anch = torch.sigmoid(head_anchor_output['weights'])
            logP_anch = head_anchor_output.get('log_probs', None)
            logP = (logP + logP_anch) / 2
            del head_anchor_output
        else:
            flow_anch, weight_anch = None, None

        rewards, pred_trans = self.compute_reward(
            action_points, anchor_points, flow_act, weight_act, flow_anch, weight_anch, gt_trans)
        rewards = rewards.reshape(self.group, -1)
        std, mean = torch.std_mean(rewards, dim=0, keepdim=True)
        adv = (rewards - mean) / (std + 1e-8)
        return {
            "state_act": action_points,
            "state_anch": anchor_points,
            "act_1": flow_act,
            "weight_act": weight_act,
            "act_2": flow_anch,
            "weight_anch": weight_anch,
            "reward": rewards,
            "pred_trans": pred_trans,
            "reward_std": std,
            "reward_mean": mean,
            "adv": adv,
            "log_prob": logP,
            "cache": (
                action_points, anchor_points,
                action_embedding, anchor_embedding,
                action_pt_pos, anchor_pt_pos,
                act_down_sample, anch_down_sample
            ) if return_cache else None
        }

    def log_probs(self, action_points, anchor_points, flow_act, flow_anch,
                  cache=None):
        if cache is None:
            (
                action_points, anchor_points,
                action_embedding, anchor_embedding,
                action_pt_pos, anchor_pt_pos,
                act_down_sample, anch_down_sample
            ) = self._embedding(action_points, anchor_points)
        else:
            (
                action_points, anchor_points,
                action_embedding, anchor_embedding,
                action_pt_pos, anchor_pt_pos,
                act_down_sample, anch_down_sample
            ) = cache
        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn
        ) = self._backbone(action_embedding, anchor_embedding, action_pt_pos, anchor_pt_pos)
        assert action_attn is not None, "Attention scores are required for log_prob computation"
        logP = self.head_action.log_probs(
            action_embedding_tf,
            action_embedding,
            anchor_embedding,
            action_points,
            anchor_points,
            act_down_sample,
            anch_down_sample,
            scores=action_attn,
            actions=flow_act
        )
        if self.cycle:
            logP_anch = self.head_anchor.log_probs(
                anchor_embedding_tf,
                anchor_embedding,
                action_embedding,
                anchor_points,
                action_points,
                anch_down_sample,
                act_down_sample,
                scores=anchor_attn,
                actions=flow_anch
            )
            logP = (logP_anch + logP) / 2

        return logP


class SE3PolicyModel(ResidualFlow_DiffEmbTransformer):
    """基于 SE(3) 变换参数化的 PolicyModel。

    与 PolicyModel 不同:
      - PolicyModel 在逐点流动 (flow) 空间采样, 再通过加权 SVD 求解 SE(3) 变换
      - SE3PolicyModel 先从 flow 求解均值 SE(3) 变换 (四元数 + 平移),
        然后在 SE(3) 参数空间直接建立分布并采样

    网络结构:
      ResidualFlow_DiffEmbTransformer (base model, 输出 flow)
        → flow2pose / dualflow2pose (求解均值变换)
        → translate_var / rotate_var (从 embedding 预测方差)
        → MultivariateNormal 采样 (四元数 + 平移)
    """

    def __init__(
        self,
        encoder_cfg,
        head_cfg,
        reward_model_path=None,
        base_model_path=None,
        cycle=True,
        center_feature=False,
        freeze_embnn=False,
        return_attn=True,
        multilaterate=False,
        mlat_sample: bool = False,
        mlat_nkps: int = 100,
        feature_channels=0,
        conditional: bool = False,
        dropout=0.1,
        pos_encoding=False,
        group=32,
        n_blocks=1,
        attn_mode="torch_attn",
        manual_reawrd=False,
    ):
        assert head_cfg.head_type in ["residual", "transformer"], \
            "SE3PolicyModel only supports residual and transformer head type"
        super(SE3PolicyModel, self).__init__(
            encoder_cfg,
            head_cfg,
            cycle,
            center_feature,
            freeze_embnn,
            return_attn,
            multilaterate,
            mlat_sample,
            mlat_nkps,
            feature_channels,
            conditional,
            dropout,
            pos_encoding,
            n_blocks,
            attn_mode=attn_mode
        )
        assert reward_model_path is not None or manual_reawrd == False
        if not manual_reawrd:
            self.reward_model = RewardModel(
                encoder_cfg, cycle, False, feature_channels, dropout, False)
            self.reward_load_state_dict(reward_model_path)
            self.reward_model.eval()
        if base_model_path is not None:
            base_model_dict = torch.load(base_model_path)["state_dict"]
            base_model_dict = {k.removeprefix("model."): v for k, v in base_model_dict.items()}
            self.load_state_dict(base_model_dict, strict=True)
        # str(self.reward_model).removeprefix()
        emb_dims = encoder_cfg.emb_dims
        # 方差预测头: pool 到全局 → MLP → 直接输出全局 log-variance
        self.translate_var = nn.Sequential(
            nn.Linear(emb_dims * 2, emb_dims),
            nn.LayerNorm(emb_dims),
            nn.GELU(),
            nn.Linear(emb_dims, emb_dims // 2),
            nn.LayerNorm(emb_dims // 2),
            nn.GELU(),
            nn.Linear(emb_dims // 2, 3),       # (B, 3)
        )
        self.rotate_var = nn.Sequential(
            nn.Linear(emb_dims * 2, emb_dims),
            nn.LayerNorm(emb_dims),
            nn.GELU(),
            nn.Linear(emb_dims, emb_dims // 2),
            nn.LayerNorm(emb_dims // 2),
            nn.GELU(),
            nn.Linear(emb_dims // 2, 3),       # (B, 3) — so(3) 切空间
        )
        self.manual_reawrd = manual_reawrd
        self.group = group

    def get_param_groups(self, base_lr: float, var_lr_mult: float = 10.0):
        """返回优化器参数组。

        translate_var / rotate_var 使用更高学习率 (var_lr_mult * base_lr),
        因为它们未经预训练, 需要更快收敛.
        """
        var_params = (list(self.translate_var.parameters()) +
                      list(self.rotate_var.parameters()))
        var_ids = {id(p) for p in var_params}
        base_params = [p for p in self.parameters() if id(p) not in var_ids]
        return [
            {"params": base_params, "lr": base_lr},
            {"params": var_params,  "lr": base_lr * var_lr_mult},
        ]

    # ---- helpers ----

    @staticmethod
    def _extract_flow_and_weight(full_flow: torch.Tensor):
        """从 full_flow (B, N, 4) 中分离 flow 和 weight."""
        pred_flow = full_flow[:, :, :3]           # (B, N, 3)
        pred_w = torch.sigmoid(full_flow[:, :, 3])  # (B, N)
        return pred_flow, pred_w

    def _flow_to_transform(self,
                           action_pts, anchor_pts,
                           flow_action, flow_anchor=None):
        """将 flow 转换为 Transform3d (均值变换)."""
        if flow_anchor is None:
            pred_flow, pred_w = self._extract_flow_and_weight(flow_action)
            return flow2pose(
                xyz=action_pts, flow=pred_flow, weights=pred_w,
                return_transform3d=True), (pred_w, None)
        else:
            pred_flow_a, pred_w_a = self._extract_flow_and_weight(flow_action)
            pred_flow_b, pred_w_b = self._extract_flow_and_weight(flow_anchor)
            return dualflow2pose(
                xyz_src=action_pts, xyz_tgt=anchor_pts,
                flow_src=pred_flow_a, flow_tgt=pred_flow_b,
                weights_src=pred_w_a, weights_tgt=pred_w_b,
                return_transform3d=True, training=True), (pred_w_a, pred_w_b)

    @staticmethod
    def _global_pool(emb_per_pt: torch.Tensor, weights=None) -> torch.Tensor:
        """逐点 embedding (B, C, N) → 全局 embedding (B, C)."""
        if weights is None:
            return emb_per_pt.mean(dim=-1)
        else:
            return (emb_per_pt * weights.unsqueeze(1)).sum(dim=-1) / weights.sum(dim=-1, keepdim=True)

    def _build_transform_from_samples(self, quat_samples, t_samples):
        """从采样的四元数和平移构造 Transform3d.

        Args:
            quat_samples: (G, B, 4)  四元数 (w,x,y,z)
            t_samples:    (G, B, 3)  平移向量
        Returns:
            sample_trans: Transform3d with batch (G*B,)
        """
        G, B, _ = quat_samples.shape
        quat_flat = F.normalize(quat_samples.reshape(-1, 4), dim=-1)
        t_flat = t_samples.reshape(-1, 3)
        R_flat = quaternion_to_matrix(quat_flat)          # (G*B, 3, 3)
        mat = torch.eye(4, device=quat_samples.device,
                        dtype=quat_samples.dtype
                        ).unsqueeze(0).expand(G * B, -1, -1).clone()
        mat[:, :3, :3] = R_flat
        mat[:, :3, 3] = t_flat
        return Transform3d(matrix=mat)

    # ---- so(3) 切空间采样 (替代 4D 独立高斯) ----

    @staticmethod
    def _sample_quaternion_tangent(quat_mean, rot_logvar, group):
        """在 so(3) 切空间采样, 通过指数映射到 S³.

        原理:
          - 四元数在 S³ 上, 只有 3 个自由度
          - 在均值 q_mean 处的切空间是 so(3) ≅ ℝ³
          - 在 ℝ³ 中采样 ε ~ N(0, diag(σ²)), 然后用指数映射:
            δq = (cos(|ε|/2), sin(|ε|/2) · ε/|ε|)  天然在 S³ 上
          - 最终: q_sample = q_mean ⊗ δq

        Args:
            quat_mean:  (B, 4)  均值四元数 (w,x,y,z)
            rot_logvar: (B, 3)  so(3) 切空间对数方差
            group:      int      每组采样数
        Returns:
            quat_samples: (G, B, 4)  采样四元数
            log_prob:     (G, B)     切空间对数概率
        """
        B = quat_mean.shape[0]
        device = quat_mean.device
        eps = 1e-8

        # 1. 在 ℝ³ 切空间采样
        rot_dist = MultivariateNormal(
            torch.zeros(3, device=device),
            torch.diag_embed(rot_logvar.exp()))
        epsilon = rot_dist.sample((group,))                 # (G, B, 3)
        log_prob = rot_dist.log_prob(epsilon)               # (G, B)

        # 2. 指数映射 ℝ³ → S³ (轴角 → 四元数)
        theta = epsilon.norm(p=2, dim=-1, keepdim=True)     # (G, B, 1)
        safe_theta = torch.where(theta < eps, torch.ones_like(theta), theta)
        axis = epsilon / safe_theta                          # (G, B, 3)

        cos_half = torch.cos(theta / 2.0)
        sin_half = torch.sin(theta / 2.0)
        delta_q = torch.cat([cos_half, sin_half * axis], dim=-1)  # (G, B, 4)

        # 3. 左乘均值四元数
        G = group
        quat_mean_exp = quat_mean.unsqueeze(0).expand(G, -1, -1)
        quat_samples = quaternion_multiply(
            quat_mean_exp.reshape(-1, 4),
            delta_q.reshape(-1, 4),
        ).reshape(G, B, 4)

        return quat_samples, log_prob

    # ---- reward functions ----
    def ideal_reward(self,
                     pred_trans: Transform3d, s_next,
                     act_pts, anchor_pts,
                     gt_trans: Transform3d,
                     s_next_inv=None):
        """SE3 版本的 ideal_reward: point_loss + pose_error.

        与 PolicyModel.ideal_reward 不同: 没有 smoothness/dense 分量,
        因为 SE3PolicyModel 直接采样刚体变换, 不存在 flow 自洽性问题。

        当 s_next_inv 非 None 时 (cycle=True), 额外计算 anchor 侧的
        point_loss 和 pose_error (GT 为 gt_T^{-1}), 两侧取平均。
        """
        bz = gt_trans.get_matrix().shape[0]
        gn = act_pts.shape[0] // bz
        gt_trans_mat = gt_trans.get_matrix().unsqueeze(0).expand(
            gn, -1, -1, -1).reshape(-1, 4, 4)
        gt_trans_group = Transform3d(matrix=gt_trans_mat)

        # --- action 侧 ---
        # gt_transformed_pts = gt_trans_group.transform_points(act_pts)
        # point_dist_a = (s_next - gt_transformed_pts).norm(p=2, dim=-1).max(dim=-1)[0]
        # r_point_a = torch.clamp_max(1.0 / (point_dist_a + 1e-6), max=1e6)

        error_mat = gt_trans_group.compose(pred_trans.inverse())
        error_R_a = get_degree_angle(error_mat, return_batch=True)[0]
        error_t_a = get_translation(error_mat, return_batch=True)[0]
        r_pose_a = torch.clamp_max(1.0 / (error_R_a + error_t_a + 1e-6), max=1e6)

        r_point = r_pose_a # + r_pose_a

        # --- anchor 侧 (cycle 模式) ---
        if s_next_inv is not None:
            # gt_inv_pts = gt_trans_group.inverse().transform_points(anchor_pts)
            # point_dist_b = (s_next_inv - gt_inv_pts).norm(p=2, dim=-1).max(dim=-1)[0]
            # r_point_b = torch.clamp_max(1.0 / (point_dist_b + 1e-6), max=1e6)

            # pred_trans.inverse() 是 pred_T_anchor, GT 是 gt_T.inverse()
            error_mat_b = gt_trans_group.inverse().compose(pred_trans)
            error_R_b = get_degree_angle(error_mat_b, return_batch=True)[0]
            error_t_b = get_translation(error_mat_b, return_batch=True)[0]
            r_pose_b = torch.clamp_max(1.0 / (error_R_b + error_t_b + 1e-6), max=1e6)

            # r_point = (r_point_a + r_point_b) / 2.0
            r_point = (r_pose_a + r_pose_b) / 2.0

        return r_point # + r_pose

    @torch.no_grad()
    def compute_reward(self,
                       act_pts, anchor_pts,
                       quat_samples, t_samples,
                       gt_trans=None):
        """计算 SE3 采样的 reward。

        Args:
            act_pts:     (B, 3, N)
            anchor_pts:  (B, 3, N)
            quat_samples: (G, B, 4)
            t_samples:    (G, B, 3)
            gt_trans:     Transform3d (B,)
        Returns:
            rewards:       (G*B,) 标量奖励
            sample_trans:  Transform3d (G*B,)
        """
        _, _, n = act_pts.shape
        Gn, _, _ = quat_samples.shape
        # expand 点云到 group 维度
        act_pts_exp = act_pts.permute(0, 2, 1).unsqueeze(0).expand(
            Gn, -1, -1, -1).contiguous().reshape(-1, n, 3)
        anchor_pts_exp = anchor_pts.permute(0, 2, 1).unsqueeze(0).expand(
            Gn, -1, -1, -1).contiguous().reshape(-1, n, 3)

        # 构造 sample 变换
        sample_trans = self._build_transform_from_samples(
            quat_samples, t_samples)
        s_next = sample_trans.transform_points(act_pts_exp)

        # anchor 侧: T^{-1} 变换 anchor 点云
        s_next_inv = None
        if self.cycle:
            s_next_inv = sample_trans.inverse().transform_points(
                anchor_pts_exp)

        if self.manual_reawrd:
            assert gt_trans is not None
            rewards = self.ideal_reward(
                sample_trans, s_next,
                act_pts_exp, anchor_pts_exp, gt_trans,
                s_next_inv=s_next_inv)
        else:
            rewards = self.reward_model(
                s_next, anchor_pts_exp, return_total_reward=True)
            if self.cycle:
                rewards_inv = self.reward_model(
                    s_next_inv, act_pts_exp, return_total_reward=True)
                rewards = (rewards + rewards_inv) / 2.0
        return rewards, sample_trans

    # ---- RL 核心方法 ----

    @torch.no_grad()
    def rl_sample(self, *input, return_cache=False, gt_trans=None):
        """执行一次采样: embedding → backbone → flow2pose → 分布采样 → reward.

        Returns 字典 (键名与 PolicyModel.rl_sample 尽量兼容):
          quat_samples:  (G, B, 4)  采样的四元数
          t_samples:     (G, B, 3)  采样的平移
          quat_mean:     (B, 4)     均值四元数
          t_mean:        (B, 3)     均值平移
          trans_logvar:  (B, 3)     平移对数方差
          rot_logvar:    (B, 4)     旋转对数方差
          reward:        (G, B)     奖励
          pred_trans_mean: Transform3d (B,) 均值变换
          reward_std:    (1, B)
          reward_mean:   (1, B)
          adv:           (G, B)     advantage
          log_prob:      (G, B)     log 概率
          cache:         嵌入缓存 (可选)
        """
        # 1. embedding + backbone
        (
            action_points, anchor_points,
            action_embedding, anchor_embedding,
            action_pt_pos, anchor_pt_pos,
            act_down_sample, anch_down_sample
        ) = self._embedding(*input)

        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn
        ) = self._backbone(
            action_embedding, anchor_embedding,
            action_pt_pos, anchor_pt_pos)

        # 2. 确定性 head forward → flow → 均值 SE(3)
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
        flow_action = head_action_output["full_flow"].permute(
            0, 2, 1).contiguous()              # (B, N, 4)

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
            flow_anchor = head_anchor_output["full_flow"].permute(
                0, 2, 1).contiguous()           # (B, N, 4)
        else:
            flow_anchor = None

        # action 点云坐标 (B, N, 3) 用于 flow2pose
        action_pts_xyz = action_points.permute(0, 2, 1).contiguous()[:, :, :3]
        anchor_pts_xyz = anchor_points.permute(0, 2, 1).contiguous()[:, :, :3]

        pred_trans_mean, (pred_w_a, pred_w_b) = self._flow_to_transform(
            action_pts_xyz, anchor_pts_xyz,
            flow_action, flow_anchor)

        # 3. 提取均值参数
        R_mean = pred_trans_mean.get_matrix()[:, :3, :3]   # (B, 3, 3)
        quat_mean = matrix_to_quaternion(R_mean)            # (B, 4)
        t_mean = pred_trans_mean.get_matrix()[:, :3, 3]     # (B, 3)

        # 4. 从 embedding 预测全局方差
        pool_action_embedding = self._global_pool(
            action_embedding_tf, pred_w_a)
        pool_anchor_embedding = self._global_pool(
            anchor_embedding_tf, pred_w_b if self.cycle else None)
        global_emb = torch.cat(
            [pool_action_embedding, pool_anchor_embedding], dim=1)  # (B, 2*emb)
        # global_emb = self._global_pool(concat_emb)              # (B, 2*emb)

        trans_logvar = self.translate_var(global_emb)            # (B, 3)
        rot_logvar = self.rotate_var(global_emb)                 # (B, 3) — so(3) 切空间

        # 5. 构建分布并采样
        trans_dist = MultivariateNormal(
            t_mean, torch.diag_embed(trans_logvar.exp()))

        t_samples = trans_dist.sample((self.group,))                      # (G, B, 3)
        quat_samples, logP_r_sampled = self._sample_quaternion_tangent(
            quat_mean, rot_logvar, self.group)                            # (G, B, 4), (G, B)

        # ### DEBUG: 计算 base model预测均值的reward
        # rewards, _ = self.compute_reward(
        #     action_points, anchor_points,
        #     quat_mean.unsqueeze(0), t_mean.unsqueeze(0), gt_trans)
        # print(f"Debug SE3PolicyModel: reward at mean = {rewards.mean().item():.4f}")

        # 6. 计算 reward
        rewards, _ = self.compute_reward(
            action_points, anchor_points,
            quat_samples, t_samples, gt_trans)
        rewards = rewards.reshape(self.group, -1)               # (G, B)
        std, mean = torch.std_mean(rewards, dim=0, keepdim=True)
        adv = (rewards - mean) / (std + 1e-8)

        # 7. 计算 log 概率
        logP_t = trans_dist.log_prob(t_samples)                 # (G, B)
        logP = (logP_t + logP_r_sampled) / 2.0

        return {
            "act_1": quat_samples,
            "act_2": t_samples,
            "quat_mean": quat_mean,
            "t_mean": t_mean,
            "trans_logvar": trans_logvar,
            "rot_logvar": rot_logvar,
            "reward": rewards,
            "pred_trans_mean": pred_trans_mean,
            "reward_std": std,
            "reward_mean": mean,
            "adv": adv,
            "log_prob": logP,
            "cache": (
                action_points, anchor_points,
                action_embedding, anchor_embedding,
                action_pt_pos, anchor_pt_pos,
                act_down_sample, anch_down_sample
            ) if return_cache else None,
        }

    def log_probs(self, action_points, anchor_points,
                  quat_actions, t_actions,
                  cache=None):
        """计算给定 SE3 动作在当前策略下的对数概率.

        Args:
            action_points: (B, 3, N)
            anchor_points: (B, 3, N)
            quat_actions:  (G, B, 4)  采样的四元数
            t_actions:     (G, B, 3)  采样的平移
            cache:         嵌入缓存
        Returns:
            logP: (G, B)
        """
        if cache is None:
            (
                action_points, anchor_points,
                action_embedding, anchor_embedding,
                action_pt_pos, anchor_pt_pos,
                act_down_sample, anch_down_sample
            ) = self._embedding(action_points, anchor_points)
        else:
            (
                action_points, anchor_points,
                action_embedding, anchor_embedding,
                action_pt_pos, anchor_pt_pos,
                act_down_sample, anch_down_sample
            ) = cache

        (
            action_embedding_tf,
            anchor_embedding_tf,
            action_attn,
            anchor_attn
        ) = self._backbone(
            action_embedding, anchor_embedding,
            action_pt_pos, anchor_pt_pos)

        # 确定性 forward → 均值 SE(3)
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
        flow_action = head_action_output["full_flow"].permute(0, 2, 1)
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
            flow_anchor = head_anchor_output["full_flow"].permute(0, 2, 1)
        action_pts_xyz = action_points.permute(0, 2, 1).contiguous()[:, :, :3]
        anchor_pts_xyz = anchor_points.permute(0, 2, 1).contiguous()[:, :, :3]
        pred_trans_mean, _ = self._flow_to_transform(
            action_pts_xyz, anchor_pts_xyz, flow_action, flow_anchor if self.cycle else None)

        # 均值参数
        R_mean = pred_trans_mean.get_matrix()[:, :3, :3]
        quat_mean = matrix_to_quaternion(R_mean)
        t_mean = pred_trans_mean.get_matrix()[:, :3, 3]

        # 方差
        concat_emb = torch.cat(
            [action_embedding_tf, anchor_embedding_tf], dim=1)
        global_emb = self._global_pool(concat_emb, )

        trans_logvar = self.translate_var(global_emb)
        rot_logvar = self.rotate_var(global_emb)

        # 计算给定动作的 log prob
        trans_dist = MultivariateNormal(
            t_mean, torch.diag_embed(trans_logvar.exp()))
        logP_t = trans_dist.log_prob(t_actions)   # (G, B)

        # so(3) 切空间: 反推 quat_actions 对应的 ε
        # q_act = q_mean ⊗ δq  →  δq = conj(q_mean) ⊗ q_act
        # δq = (cos(θ/2), sin(θ/2)·axis) → ε = θ · axis
        quat_mean_inv = quat_mean * torch.tensor(
            [1, -1, -1, -1], device=quat_mean.device, dtype=quat_mean.dtype)
        G = quat_actions.shape[0]
        delta_q = quaternion_multiply(
            quat_mean_inv.unsqueeze(0).expand(G, -1, -1).reshape(-1, 4),
            quat_actions.reshape(-1, 4),
        ).reshape(G, -1, 4)                                     # (G, B, 4)

        cos_half = delta_q[..., 0:1].clamp(-1.0, 1.0)
        theta = 2.0 * torch.acos(cos_half)                      # (G, B, 1)
        sin_part = delta_q[..., 1:4]                            # (G, B, 3)
        sin_norm = sin_part.norm(p=2, dim=-1, keepdim=True) + 1e-8
        axis = sin_part / sin_norm
        epsilon = theta * axis                                  # (G, B, 3)

        rot_dist = MultivariateNormal(
            torch.zeros(3, device=quat_mean.device),
            torch.diag_embed(rot_logvar.exp()))
        logP_r = rot_dist.log_prob(epsilon)                     # (G, B)
        return (logP_t + logP_r) / 2.0


def create_policy_model(cfg):
    """创建策略模型.

    Args:
        cfg: 模型配置
    Returns:
        nn.Module
    """
    if cfg.model.model_type == "rl_flow":
        return PolicyModel(
            cfg.model.encoder,
            cfg.model.head,
            cfg.rl.reward_model_path,
            cfg.model.cycle,
            center_feature=cfg.model.center_feature,
            freeze_embnn=cfg.model.freeze_embnn,
            return_attn=cfg.model.return_attn,
            dropout=cfg.model.dropout,
            pos_encoding=cfg.model.pos_encoding,
            group=cfg.rl.group,
            n_blocks=int(cfg.model.n_blocks),
            attn_mode=cfg.model.attn_mode,
            manual_reawrd=True
        )
    elif cfg.model.model_type == "rl_se3":
        return SE3PolicyModel(
            cfg.model.encoder,
            cfg.model.head,
            cfg.rl.reward_model_path,
            cfg.rl.base_model_path,
            cfg.model.cycle,
            center_feature=cfg.model.center_feature,
            freeze_embnn=cfg.model.freeze_embnn,
            return_attn=cfg.model.return_attn,
            dropout=cfg.model.dropout,
            pos_encoding=cfg.model.pos_encoding,
            group=cfg.rl.group,
            n_blocks=int(cfg.model.n_blocks),
            attn_mode=cfg.model.attn_mode,
            manual_reawrd=True
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.model_type}")


if __name__ == "__main__":
    from taxpose.nets.raw_dgcnn import DGCNNArgs
    from taxpose.nets.head import HeadConfig

    p1 = torch.randn(2, 512, 3)
    p2 = torch.randn(2, 512, 3)
    encoder_args = DGCNNArgs(
        name="raw_dgcnn",
        emb_dims=512,
        knn=2,
    )
    model = PolicyModel(
        encoder_args,
        HeadConfig(
            head_type="rl_residual",
            emb_dims=512,
            project_corrs=True,
            output_num=512
        ),
        reward_model_path="/home/yan/pose_estimation/taxpose/logs/rl_reward/2026-05-24/10-51-49/checkpoints/last.ckpt",
        cycle=True,
        center_feature=True,
        dropout=0.1
    )
    y = model(p1, p2)
    print(f"forward: {y.keys()}")
    samples = model.rl_sample(p1, p2, return_cache=True)
    print(f"sample: {samples.keys()}")
    print(f"sample: {samples['flow_act'].shape}")
    print(f"sample: {samples['adv'].shape}")
    print(f"cache: {samples['cache'][0].shape}")
    log_p = model.log_probs(p1, p2, samples['flow_act'], samples['flow_anch'], cache=samples['cache'])
    print(f"shape: {log_p.shape}")

