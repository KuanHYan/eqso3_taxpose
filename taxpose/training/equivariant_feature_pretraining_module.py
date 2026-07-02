import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pytorch3d.transforms import Rotate, random_rotations
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import ToTensor

from taxpose.training.point_cloud_training_module import PointCloudTrainingModule
from taxpose.utils.emb_losses import (
    dist2weight,
    infonce_loss,
    mean_geo_diff,
    mean_order,
)

mse_criterion = nn.MSELoss(reduction="sum")
to_tensor = ToTensor()


class EquivariancePreTrainingModule(PointCloudTrainingModule):
    def __init__(
        self,
        model=None,
        lr=1e-3,
        image_log_period=500,
        l2_reg_weight=0.00,
        normalize_features=True,
        temperature=0.1,
        con_weighting="dist",
        lr_cfg: dict = {'scheduler': 'constant', 'max_steps': 100, 'warmup_ratio': 0.0, 'by_epoch': True},
        debug=False,
        optimization_mode: str = 'auto'
    ):
        super().__init__(
            model=model,
            lr=lr,
            image_log_period=image_log_period,
            debug=debug,
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

    def similarity_geo_distance(self, similarity, points):
        test_idx = np.random.randint(similarity.shape[1])
        dist = (points[0] - points[0, test_idx].unsqueeze(0)).norm(dim=-1)
        fig = plt.figure(figsize=(10, 7.5))

        ax_sim = fig.add_subplot(111)

        ax_sim.scatter(
            dist.detach().cpu().numpy(),
            similarity[0, test_idx].detach().cpu().numpy(),
        )
        ax_sim.set_ylabel("Similarity")
        ax_sim.set_xlabel("Geometric Distance")

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img = np.array(canvas.buffer_rgba())
        plt.close(fig)
        return img

    def module_step(self, batch, batch_idx):
        points = batch  # B, num_points, 3
        transforms = Rotate(random_rotations(len(points), device=points.device))
        points_centered = points - points.mean(dim=1, keepdims=True)  # B, num_points, 3

        points_trans = transforms.transform_points(points)
        points_trans_centered = points_trans - points_trans.mean(dim=1, keepdims=True)
        # B, num_points, 3

        phi, pts = self.model(points_centered.transpose(-1, -2))
        phi_trans, pts_trans = self.model(points_trans_centered.transpose(-1, -2))

        # ****************DEBUG***********************
        # 在训练循环中加入监控
        debug_log_values = {}
        debug_log_values.update(
            phi_mean=phi.abs().mean().item(),
            phi_std=phi.std().item(),
            phi_nonzero_ratio=(phi != 0).float().mean().item())

        if self.normalize_features:
            phi = F.normalize(phi, dim=1)
            phi_trans = F.normalize(phi_trans, dim=1)

            debug_log_values.update(
                phi_norm_mean=phi.abs().mean().item(),
                phi_norm_std=phi.std().item(),
                phi_norm_nonzero_ratio=(phi != 0).float().mean().item())

        if points.shape[1] != phi.shape[2]:
            points = pts

        if self.con_weighting.lower() == "mask":
            w = dist2weight(points_centered)
        elif self.con_weighting.lower() == "dist":
            w = dist2weight(points, func=lambda x: torch.tanh(10 * x))
        else:
            w = None

        contrastive_loss, similarity = infonce_loss(
            phi, phi_trans, weights=w, temperature=self.temperature
        )

        loss = contrastive_loss

        mean_order_error = mean_order(similarity)
        mean_geo_error = mean_geo_diff(similarity, points)

        log_values = {}
        log_values["contrastive_loss"] = contrastive_loss
        # log_values['loss'] = loss
        log_values["mean_geo_diff"] = mean_geo_error
        log_values["mean_order"] = mean_order_error

        if self.l2_reg_weight > 0:
            phi_norm = phi.norm(dim=1, keepdim=True)
            phi_trans_norm = phi_trans.norm(dim=1, keepdim=True)
            l2_reg = mse_criterion(
                phi_norm, torch.zeros_like(phi_norm)
            ) + mse_criterion(phi_trans_norm, torch.zeros_like(phi_trans_norm))

            loss = loss + self.l2_reg_weight * l2_reg
            log_values["l2_reg_loss"] = self.l2_reg_weight * l2_reg

        log_values.update(debug_log_values)
        return loss, log_values

    def visualize_results(self, batch, batch_idx):
        points = batch  # B, num_points, 3
        transforms = Rotate(random_rotations(len(points), device=points.device))

        points_centered = points - points.mean(dim=1, keepdims=True)  # B, num_points, 3
        points_trans = transforms.transform_points(points)
        points_trans_centered = points_trans - points_trans.mean(dim=1, keepdims=True)

        # 提取特征，形状：[B, C, N]
        phi, pts = self.model(points_centered.transpose(-1, -2))
        phi_trans, _ = self.model(points_trans_centered.transpose(-1, -2))
        if self.normalize_features:
            phi = F.normalize(phi, dim=1)
            phi_trans = F.normalize(phi_trans, dim=1)

        # 相似度矩阵 [B, N, N]
        similarity = phi.transpose(-1, -2) @ phi_trans

        # ---------- PCA 提取前三个主成分作为颜色 ----------
        def pca_colors(feature_tensor):
            """
            feature_tensor: [B, C, N] 取第一个 batch -> [N, C]
            返回: [N, 3] 的 RGB 颜色数组，值域 [0,255]
            """
            feat = feature_tensor[0].detach().cpu().numpy()  # [C, N]
            feat = feat.T  # [N, C]
            # 若特征维度小于3，则补零或直接使用（这里假设C>=3）
            n_components = min(3, feat.shape[1])
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(feat)  # [N, n_components]
            if n_components < 3:
                # 补零至3维
                pca_result = np.pad(pca_result, ((0,0),(0,3-n_components)), mode='constant')
            # 对每列做 min-max 归一化到 [0,1]
            pca_min = pca_result.min(axis=0, keepdims=True)
            pca_max = pca_result.max(axis=0, keepdims=True)
            pca_norm = (pca_result - pca_min) / (pca_max - pca_min + 1e-8)
            # 映射到 0-255
            colors = (pca_norm * 255)
            return pca_norm, colors

        phi_pca, color = pca_colors(phi)          # [N, 3]
        phi_trans_pca,color_trans = pca_colors(phi_trans)  # [N, 3]

        # 合并点云坐标与颜色，用于 wandb 3D 可视化
        if points.shape[1] != phi.shape[2]:
            points = pts
        points_np = points[0].detach().cpu().numpy()
        points_emb = np.concatenate([points_np, color], axis=-1)
        points_trans_emb = np.concatenate([points_np, color_trans], axis=-1)

        # ---------- 相似度颜色映射（归一化 + colormap） ----------
        diag_sim = similarity[0].diagonal().detach().cpu().numpy()  # [N]
        sim_min = diag_sim.min()
        sim_max = diag_sim.max()
        sim_norm = (diag_sim - sim_min) / (sim_max - sim_min + 1e-8)
        # 使用 coolwarm (蓝-白-红) 映射
        color_dist = (255 * cm.coolwarm(sim_norm)[:, :3])  # [N, 3]

        # 构建对比显示的点云（相似度颜色）
        points_comp_disp = np.concatenate([points_np, color_dist], axis=-1)

        # 组装返回字典
        res_viz = {
            "points_emb": wandb.Object3D(points_emb),
            "points_trans_emb": wandb.Object3D(points_trans_emb),
            "points_comp_disp": wandb.Object3D(points_comp_disp),
            "sim_geo_distance": self.similarity_geo_distance(similarity, points)
        }
        return res_viz
