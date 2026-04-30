import torch
import wandb
from pytorch3d.transforms import Rotate, random_rotations
from pytorch3d.loss import chamfer_distance
from torch import nn
from torchvision.transforms import ToTensor

from taxpose.training.point_cloud_training_module import PointCloudTrainingModule
from taxpose.utils.color_utils import get_color
from taxpose.nets.points_vae import PointNet2AutoEncoder

to_tensor = ToTensor()


class EquivariancePreTrainingModule(PointCloudTrainingModule):
    def __init__(
        self,
        model: PointNet2AutoEncoder,
        lr=1e-3,
        image_log_period=500,
        l2_reg_weight=0.00,
        normalize_features=True,
        temperature=0.1,
        con_weighting="dist",
        lr_cfg: dict = {'scheduler': 'constant', 'max_steps': 100, 'warmup_ratio': 0.0, 'by_epoch': True},
        tensorboard_writer=None,
        optimization_mode: str = 'auto'
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
        self.l2_reg_weight = l2_reg_weight
        self.normalize_features = normalize_features
        self.temperature = temperature
        self.con_weighting = con_weighting

    def module_step(self, batch, batch_idx):
        points = batch  # B, num_points, 3
        transforms = Rotate(random_rotations(len(points), device=points.device)).translate(...)
        points_centered = points - points.mean(dim=1, keepdims=True)  # B, num_points, 3

        points_trans = transforms.transform_points(points)
        points_trans_centered = points_trans - points_trans.mean(dim=1, keepdims=True)
        # B, num_points, 3

        coords, _ = self.model.forward(points_trans)

        loss = chamfer_distance(points, coords, )

        log_values = {}
        log_values["loss"] = ...
        # add other metrics
        log_values["..."] = ...

        return loss, log_values

    @torch.no_grad()
    def visualize_results(self, batch, batch_idx):
        points = batch  # B, num_points, 3
        transforms = Rotate(random_rotations(len(points), device=points.device)).translate(...)

        points_centered = points - points.mean(dim=1, keepdims=True)  # B, num_points, 3
        points_trans = transforms.transform_points(points)
        points_trans_centered = points_trans - points_trans.mean(dim=1, keepdims=True)

        # 提取特征，形状：[B, C, N]
        coords, _ = self.model(points_trans)

        # 合并点云坐标与颜色，用于 wandb 3D 可视化
        points_gt = points[0].detach().cpu()
        point_pred = coords[0].detach().cpu()
        points_vis = get_color([points_gt, point_pred], ['red', 'blue'])
        # 组装返回字典
        res_viz = {
            "points_comp": wandb.Object3D(points_vis),
            "other_metrics": ...
        }
        return res_viz
