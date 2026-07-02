from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import Transform3d
from pytorch3d.loss import chamfer_distance

from taxpose.training.point_cloud_training_module import PointCloudTrainingModule
from taxpose.utils.se3 import mse_criterion, random_se3
from taxpose.nets.RL_tune import RewardModel


class RLRewardTrainingModule(PointCloudTrainingModule):
    def __init__(
        self,
        model: RewardModel,
        lr=1e-3,
        lr_cfg: dict = {
            "scheduler": "constant",
            "max_steps": 400,
            "warmup_ratio": 0.1,
            "min_lr": 1e-5,
            "by_epoch": True,
        },
        sigmoid_on=False,
        point_cloud_loss: str = "CD",
        debug=False,
        optimization_mode: str = "auto",
    ):
        super().__init__(
            model=model,
            lr=lr,
            debug=debug,
            optimization_mode=optimization_mode,
            **lr_cfg,
        )
        if mse_criterion.type_key != point_cloud_loss:
            mse_criterion.set_type_key(point_cloud_loss)
        self.model = model
        self.lr = lr
        self.sigmoid_on = sigmoid_on

    def module_step(self, batch, batch_idx):
        points_trans_action = batch["points_action_trans"]
        points_trans_anchor = batch["points_anchor_trans"]
        gt_action = batch["points_action"]
        gt_anchor = batch["points_anchor"]
        T0 = Transform3d(matrix=batch["T0"])
        T1 = Transform3d(matrix=batch["T1"])

        bz, n, _ = points_trans_action.shape
        device = points_trans_action.device
        with torch.no_grad():
            transforms_win = random_se3(bz, np.pi, 0.05, device=device).inverse().compose(T1)
            pts_win = transforms_win.transform_points(points_trans_action)
            chamfer_dis_win = chamfer_distance(pts_win, points_trans_anchor)

            transforms_lose = random_se3(bz, np.pi, 0.05, device=device).inverse().compose(T1)
            pts_lose = transforms_lose.transform_points(points_trans_action)
            chamfer_dis_lose = chamfer_distance(pts_lose, points_trans_anchor)

            wrong_mask = chamfer_dis_win > chamfer_dis_lose
            pts_win[wrong_mask], pts_lose[wrong_mask] = \
                pts_lose[wrong_mask], pts_win[wrong_mask]

            # rondom sample ground truth
            gt_mask = torch.rand(bz, device=device) > 0.9
            pts_win[gt_mask] = T1.transform_points(gt_action)[gt_mask]
        
        win_output = self.model.forward(
            pts_win,
            points_trans_anchor,
        )
        lose_output = self.model.forward(
            pts_lose,
            points_trans_anchor,
        )

        log_values = {}
        loss, log_values = self.compute_loss(
            win_output, lose_output, batch, log_values=log_values, loss_prefix=""
        )
        return loss, log_values

    def compute_loss(self, win_output, lose_output, batch, log_values={}, loss_prefix=""):
        win_reward = 0.5*win_output["action_reward"] + 0.5*win_output["anchor_reward"]
        lose_reward = 0.5*lose_output["action_reward"] + 0.5*lose_output["anchor_reward"]
        loss = -F.logsigmoid(win_reward - lose_reward).mean()
        
        log_values[loss_prefix + "win_reward"] = win_reward.detach().mean()
        log_values[loss_prefix + "lose_reward"] = lose_reward.detach().mean()
        log_values[loss_prefix + "loss"] = loss.detach()
        return loss, log_values

    def forward(
        self,
        points_trans_action,
        points_trans_anchor,
        rot_tran: Transform3d = None,
    ) -> Any:
        if rot_tran is not None:
            points_trans_action = rot_tran.transform_points(points_trans_action)
            points_trans_anchor = rot_tran.transform_points(points_trans_anchor)
        res = self.model(points_trans_action, points_trans_anchor)
        reward = res['action_reward'] + res['anchor_reward']
        return reward * 0.5
