import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import Transform3d
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
            n_blocks
        )
        assert reward_model_path is not None or manual_reawrd == False
        if not manual_reawrd:
            self.reward_model = RewardModel(encoder_cfg, cycle, False, feature_channels, dropout, False)
            self.reward_load_state_dict(reward_model_path)
            self.reward_model.eval()
        self.manual_reawrd = manual_reawrd
        self.group = group

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

        # ---- 2. error_R + error_t 对应: 位姿误差 ----
        error_mat = gt_trans_group.compose(pred_trans.inverse())
        error_R = get_degree_angle(error_mat, return_batch=True)[0]        # 旋转误差 (度)
        error_t = get_translation(error_mat, return_batch=True)[0]         # 平移误差
        r_pose = torch.clamp_max(1.0 / (error_R + error_t + 1e-6), max=1e6)

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
        reward = r_point + r_pose + 0.5 * r_smooth + 0.5 * r_dense
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
            "flow_act": flow_act,
            "weight_act": weight_act,
            "flow_anch": flow_anch,
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

        logP = self.head_action.log_probs(
            action_embedding_tf,
            action_embedding,
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
                anchor_points,
                action_points,
                anch_down_sample,
                act_down_sample,
                scores=anchor_attn,
                actions=flow_anch
            )
            logP = (logP_anch + logP) / 2

        return logP


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

