from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from pytorch3d.ops import sample_farthest_points
from functools import partial
from taxpose.utils.occlusion_utils import ball_occlusion, plane_occlusion


def maybe_downsample(
    points: npt.NDArray[np.float32], num_points: Optional[int] = None
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    assert len(points.shape) == 3

    if num_points is None:
        return points, np.arange(points.shape[1])

    if points.shape[1] < num_points:
        # Randomly sample with replacement.
        n_missing = num_points - points.shape[1]
        missing_ixs = np.random.choice(points.shape[1], n_missing)
        missing_points = points[:, missing_ixs]

        # get the indices of the points from the original
        og_ixs = np.arange(points.shape[1])

        points = np.concatenate([points, missing_points], axis=1)

        return points, np.concatenate([og_ixs, missing_ixs])[None]

        # raise ValueError("Cannot downsample to more points than exist in the cloud.")

    points_pt, ids = sample_farthest_points(
        torch.from_numpy(points), K=num_points, random_start_point=True
    )

    points = points_pt.numpy()
    return points, ids.numpy()


@dataclass
class OcclusionConfig:
    occlusion_class: Union[int, str]

    # Ball occlusion.
    ball_occlusion: bool = True
    ball_radius: float = 0.1

    # Plane occlusion.
    plane_occlusion: bool = True
    plane_standoff: float = 0.04

    occlusion_prob: float = 0.5
    # 新增增强
    random_dropout: bool = True
    drop_prob: float = 0.1
    anisotropic_scaling: bool = True
    scale_range: Tuple[float, float] = (0.97, 1.03)
    gaussian_noise: bool = True
    noise_std: float = 0.01


def random_dropout(
    points: Union[npt.NDArray[np.float32], torch.Tensor],
    drop_prob: float = 0.1,
    min_keep: int = 1024
) -> Tuple[Union[npt.NDArray[np.float32], torch.Tensor], torch.Tensor]:
    """
    以概率 drop_prob 随机丢弃点。
    Args:
        points: (N, 3) numpy 或 torch 张量
        drop_prob: 每个点被丢弃的概率
        min_keep: 最少保留点数，低于此值则不丢弃
    Returns:
        retained_points: 保留的点
        mask: bool 张量 [N]，True 表示保留
    """
    is_numpy = isinstance(points, np.ndarray)
    if is_numpy:
        points_t = torch.from_numpy(points)
    else:
        points_t = points

    N = points_t.shape[0]
    keep_prob = 1.0 - drop_prob
    mask = torch.rand(N, device=points_t.device) < keep_prob

    # 如果保留点太少，放弃这次丢弃
    if mask.sum() < min_keep:
        mask = torch.ones(N, dtype=torch.bool, device=points_t.device)

    retained = points_t[mask]
    if is_numpy:
        retained = retained.numpy()
    return retained, mask


def anisotropic_scaling(
    points: Union[npt.NDArray[np.float32], torch.Tensor],
    scale_range: Tuple[float, float] = (0.97, 1.03),
    axes: Tuple[int, ...] = (0, 1, 2)   # 沿哪些轴缩放，默认XYZ
) -> Union[npt.NDArray[np.float32], torch.Tensor]:
    """
    沿指定轴进行各向异性的随机缩放。
    Args:
        points: (N, 3) numpy 或 torch 张量
        scale_range: 缩放因子的下限和上限
        axes: 需要缩放的轴，默认全部
    Returns:
        scaled_points: 缩放后的点云（不会移动中心）
    """
    is_numpy = isinstance(points, np.ndarray)
    if is_numpy:
        points_t = torch.from_numpy(points)
    else:
        points_t = points

    scales = torch.ones(3, device=points_t.device, dtype=points_t.dtype)
    for ax in axes:
        # 为每个指定的轴独立采样缩放因子
        s = scale_range[0] + (scale_range[1] - scale_range[0]) * torch.rand(1, device=points_t.device)
        scales[ax] = s.item()

    # 以点云中心为原点进行缩放，防止整体位移
    center = points_t.mean(dim=0, keepdim=True)
    centered = points_t - center
    scaled = centered * scales + center

    if is_numpy:
        scaled = scaled.numpy()
    return scaled


def add_gaussian_noise(
    points: Union[npt.NDArray[np.float32], torch.Tensor],
    std: float = 0.01,
    clip: Optional[float] = None
) -> Union[npt.NDArray[np.float32], torch.Tensor]:
    """
    给每个点添加独立的高斯噪声。
    Args:
        points: (N, 3) numpy 或 torch 张量
        std: 噪声标准差
        clip: 噪声的截断范围（绝对值），None 表示不截断
    Returns:
        noisy_points
    """
    is_numpy = isinstance(points, np.ndarray)
    if is_numpy:
        points_t = torch.from_numpy(points)
    else:
        points_t = points

    noise = torch.randn_like(points_t) * std
    if clip is not None:
        noise = noise.clamp(-clip, clip)

    noisy = points_t + noise
    if is_numpy:
        noisy = noisy.numpy()
    return noisy


def _occlusion_with_aug(
    points: npt.NDArray[np.float32],
    obj_class: int,
    min_num_points: int,
    cfg: OcclusionConfig
):
    # 假设 points 形状为 (1, N, 3)，先取第一个
    pc = points[0]   # (N, 3) numpy

    # 1. 遮挡增强
    if obj_class == cfg.occlusion_class or cfg.occlusion_class == "all":
        if cfg.ball_occlusion:
            if np.random.rand() < cfg.occlusion_prob:
                pc_new, _ = ball_occlusion(pc, radius=cfg.ball_radius)
                if pc_new.shape[0] > min_num_points:
                    pc = pc_new
        if cfg.plane_occlusion:
            if np.random.rand() < cfg.occlusion_prob:
                pc_new, _ = plane_occlusion(pc, stand_off=cfg.plane_standoff)
                if pc_new.shape[0] > min_num_points:
                    pc = pc_new

    # 2. 随机丢弃点
    if cfg.random_dropout:
        pc, _ = random_dropout(pc, drop_prob=cfg.drop_prob)

    # 3. 各向异性缩放
    if cfg.anisotropic_scaling:
        pc = anisotropic_scaling(pc, scale_range=cfg.scale_range)

    # 4. 高斯噪声
    if cfg.gaussian_noise:
        pc = add_gaussian_noise(pc, std=cfg.noise_std)

    # 确保返回与输入类型一致（numpy），并恢复 batch 维度
    if isinstance(pc, torch.Tensor):
        pc = pc.numpy()
    return pc[np.newaxis, :]   # (1, N, 3)


def _no_occlusion(points, obj_class, min_num_points):
    return points


def occlusion_fn(cfg: Optional[OcclusionConfig] = None):
    if cfg is None:
        return _no_occlusion          # 模块级普通函数
    return partial(_occlusion_with_aug, cfg=cfg)  # partial 可正常序列化
