import fnmatch
import functools
import os
import os.path as osp
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import random
import pickle

from taxpose.datasets.augmentations import (
    OcclusionConfig,
    maybe_downsample,
    occlusion_fn,
)
from taxpose.datasets.base import PlacementPointCloudData
from taxpose.datasets.enums import ObjectClass, Phase
from taxpose.datasets.env_mod_utils import get_random_distractor_demo  # type: ignore
# from taxpose.datasets.symmetry_utils import (
#     compute_demo_symmetry_features as new_compute_demo_symmetry_features,
# )
# from taxpose.utils.symmetry_utils import (
#     get_sym_label_pca_grasp,
#     get_sym_label_pca_place,
# )

#  0 for mug, 1 for rack, 2 for gripper
# These are the labels used in the NDF dataset
# inside demos.
OBJECT_DEMO_LABELS: Dict[ObjectClass, int] = {
    ObjectClass.MUG: 0,
    ObjectClass.RACK: 1,
    ObjectClass.GRIPPER: 2,
    ObjectClass.BOTTLE: 0,
    ObjectClass.BOWL: 0,
    ObjectClass.SLAB: 1,
}

TASK_CLASSES = Literal[ObjectClass.MUG, ObjectClass.BOTTLE, ObjectClass.BOWL]

OBJECT_LABELS_TO_CLASS: Dict[Tuple[TASK_CLASSES, int], ObjectClass] = {
    (ObjectClass.MUG, 0): ObjectClass.MUG,
    (ObjectClass.MUG, 1): ObjectClass.RACK,
    (ObjectClass.MUG, 2): ObjectClass.GRIPPER,
    (ObjectClass.BOTTLE, 0): ObjectClass.BOTTLE,
    (ObjectClass.BOTTLE, 1): ObjectClass.SLAB,
    (ObjectClass.BOTTLE, 2): ObjectClass.GRIPPER,
    (ObjectClass.BOWL, 0): ObjectClass.BOWL,
    (ObjectClass.BOWL, 1): ObjectClass.SLAB,
    (ObjectClass.BOWL, 2): ObjectClass.GRIPPER,
}


@dataclass
class CustomPointCloudDatasetConfig:
    dataset_type: ClassVar[str] = "ideal_point_cloud"
    dataset_root: Path
    dataset_indices: Optional[List[int]] = None
    num_demo: Optional[int] = None
    min_num_points: int = 1024

    cloud_type: str = "teleport"
    action_class: int = OBJECT_DEMO_LABELS[ObjectClass.MUG]
    anchor_class: int = OBJECT_DEMO_LABELS[ObjectClass.RACK]
    min_num_cameras: int = 4
    max_num_cameras: int = 4

    # Symmetry parameters.
    normalize_dist: bool = True
    object_type: ObjectClass = ObjectClass.MUG
    action: Phase = Phase.GRASP
    symmetry_after_transform: bool = False

    # Augmentation parameters.
    occlusion_cfg: Optional[OcclusionConfig] = None
    distractor_anchor_aug: bool = False
    distractor_rot_sample_method: str = "axis_angle"
    multimodal_transform_base: bool = False


# def compute_demo_symmetry_features(
#     points_action,
#     points_anchor,
#     object_type,
#     action,
#     action_class,
#     anchor_class,
#     normalize_dist,
#     skip_symmetry=False,
# ):
#     # print(
#     #     f"object_type: {object_type}, action: {action}, action_class: {action_class}, anchor_class: {anchor_class}"
#     # )
#     # Handle symmetry.
#     if skip_symmetry:
#         return None, None, None, None

#     if object_type in {ObjectClass.BOTTLE, ObjectClass.BOWL}:
#         if action == "grasp":
#             sym_dict = get_sym_label_pca_grasp(
#                 action_cloud=torch.as_tensor(points_action),
#                 anchor_cloud=torch.as_tensor(points_anchor),
#                 action_class=action_class,
#                 anchor_class=anchor_class,
#                 object_type=object_type,
#                 normalize_dist=normalize_dist,
#             )

#         elif action == "place":
#             sym_dict = get_sym_label_pca_place(
#                 action_cloud=torch.as_tensor(points_action),
#                 anchor_cloud=torch.as_tensor(points_anchor),
#                 action_class=action_class,
#                 anchor_class=anchor_class,
#                 normalize_dist=normalize_dist,
#             )

#         symmetric_cls = sym_dict["cts_cls"]  # 1, num_points
#         symmetric_cls = symmetric_cls.unsqueeze(-1).numpy()  # 1, 1, num_points

#         # We want to color the gripper somehow...
#         if action_class == OBJECT_DEMO_LABELS[ObjectClass.GRIPPER]:
#             nonsymmetric_cls = sym_dict["cts_cls_nonsym"]  # 1, num_points
#             # 1, 1, num_points
#             nonsymmetric_cls = nonsymmetric_cls.unsqueeze(-1).numpy()
#         else:
#             nonsymmetric_cls = None

#         symmetry_xyzrgb = sym_dict["fig"]
#         if action_class == 0:
#             if nonsymmetric_cls is None:
#                 nonsymmetric_cls = np.ones(
#                     (1, points_anchor.shape[1], 1), dtype=np.float32
#                 )
#             action_symmetry_features = symmetric_cls
#             anchor_symmetry_features = nonsymmetric_cls
#             action_symmetry_rgb = symmetry_xyzrgb[: points_action.shape[1], 3:][None]
#             anchor_symmetry_rgb = symmetry_xyzrgb[points_action.shape[1] :, 3:][None]
#         elif anchor_class == 0:
#             if nonsymmetric_cls is None:
#                 nonsymmetric_cls = np.ones(
#                     (1, points_action.shape[1], 1), dtype=np.float32
#                 )
#             action_symmetry_features = nonsymmetric_cls
#             anchor_symmetry_features = symmetric_cls
#             action_symmetry_rgb = symmetry_xyzrgb[points_anchor.shape[1] :, 3:][None]
#             anchor_symmetry_rgb = symmetry_xyzrgb[: points_anchor.shape[1], 3:][None]
#         else:
#             raise ValueError("this should not happen")
#     else:
#         action_symmetry_features = np.ones(
#             (1, points_action.shape[1], 1), dtype=np.float32
#         )
#         anchor_symmetry_features = np.ones(
#             (1, points_anchor.shape[1], 1), dtype=np.float32
#         )
#         action_symmetry_rgb = np.zeros((1, points_action.shape[1], 3), dtype=np.uint8)
#         anchor_symmetry_rgb = np.zeros((1, points_anchor.shape[1], 3), dtype=np.uint8)

#     return (
#         action_symmetry_features,
#         anchor_symmetry_features,
#         action_symmetry_rgb,
#         anchor_symmetry_rgb,
#     )


def is_valid_file(filename: str, min_num_points=1024) -> bool:
    """检查 npz 文件中的点云数据是否有效（无 inf/nan 且数值范围合理）"""
    def bad_points(points_np):
        if np.isinf(points_np).any() or np.isinf(points_np.mean(axis=0)).any():
            return True
        if np.abs(points_np).max() > 1e3:
            return True
        if np.isnan(points_np).any() or np.isnan(points_np.mean(axis=0)).any():
            return True
        if points_np.shape[0] < min_num_points:
            return True
        return False
    try:
        with np.load(filename, allow_pickle=True) as point_data:
            points_action_np = point_data['action']
            points_anchor_np = point_data['anchor']
            # 1. 检查是否包含 inf 或 nan
            if bad_points(points_action_np) or \
                    bad_points(points_anchor_np):
                return False
            return True
    except Exception as e:
        print(f"Error loading or checking file {filename}: {e}")
        return False


class CustomPointCloudDataset(Dataset[PlacementPointCloudData]):
    def __init__(self, cfg: CustomPointCloudDatasetConfig):
        self.dataset_root = Path(cfg.dataset_root)
        self.min_num_points = cfg.min_num_points
        self.num_demo = cfg.num_demo
        self.cloud_type = cfg.cloud_type
        self.action_class = cfg.action_class
        self.anchor_class = cfg.anchor_class
        self.min_num_cameras = cfg.min_num_cameras
        self.max_num_cameras = cfg.max_num_cameras
        self.normalize_dist = cfg.normalize_dist
        self.object_type = cfg.object_type
        self.action = cfg.action
        self.skip_symmetry = cfg.symmetry_after_transform
        self.distractor_anchor_aug = cfg.distractor_anchor_aug
        self.distractor_rot_sample_method = cfg.distractor_rot_sample_method
        self.multimodal_transform_base = cfg.multimodal_transform_base

        self.filenames = self.get_existing_data()

        if self.num_demo is not None:
            min_num = min(len(self.filenames), self.num_demo)
            self.filenames = self.filenames[: min_num]

        self.occlusion_cfg = cfg.occlusion_cfg
        self.occlusion_fn = occlusion_fn(cfg.occlusion_cfg)

    def get_existing_data(self):
        filenames = fnmatch.filter(os.listdir(self.dataset_root), f"**_asm_**.npz")
        filenames = [
            os.path.join(self.dataset_root, fn) for fn in filenames
        ]
        original_length = len(filenames)
        bad_demo_names = []
        for i in range(len(filenames)):
            filename = filenames[i]
            if not os.path.exists(filename) or \
                    not is_valid_file(filename):
                bad_demo_names.append(filename)
                continue

        # remove bad filenames in bad_demo
        if len(bad_demo_names) > 0:
            filenames = [name for name in filenames if name not in bad_demo_names]
        print(f"{len(filenames)} / {original_length} demos left")
        return filenames

    @functools.lru_cache(maxsize=100)
    def load_data(self, filename, action_class, anchor_class):
        point_data = np.load(filename, allow_pickle=True)
        
        # points_raw_np = point_data["clouds"]
        # classes_raw_np = point_data["classes"]
        # TODO: 下面代码是模拟多视角点云的随机采样，暂不需要
        # if self.min_num_cameras < 4:
        #     camera_idxs = np.concatenate(
        #         [[0], np.cumsum((np.diff(classes_raw_np) == -2))]
        #     )
        #     if not np.all(np.isin(np.arange(4), np.unique(camera_idxs))):
        #         raise ValueError(
        #             "\033[93m"
        #             + f"{filename} did not contain all classes in all cameras"
        #             + "\033[0m"
        #         )

        #     num_cameras = np.random.randint(
        #         low=self.min_num_cameras, high=self.max_num_cameras + 1
        #     )
        #     sampled_camera_idxs = np.random.choice(4, num_cameras, replace=False)
        #     valid_idxs = np.isin(camera_idxs, sampled_camera_idxs)
        #     points_raw_np = points_raw_np[valid_idxs]
        #     classes_raw_np = classes_raw_np[valid_idxs]

        points_action_np = point_data['action'].copy()
        points_action_mean_np = points_action_np.mean(axis=0)
        points_action_np = points_action_np - points_action_mean_np

        points_anchor_np = point_data['anchor'].copy()
        # points_anchor_mean_np = points_anchor_np.mean(axis=0)
        # TODO: 当前猜测应该是使用同一个均值而非使用各自的均值中心化
        points_anchor_np = points_anchor_np - points_action_mean_np

        points_action = points_action_np.astype(np.float32)[None, ...]
        points_anchor = points_anchor_np.astype(np.float32)[None, ...]

        return (points_action, points_anchor)

    def __getitem__(self, index: int) -> PlacementPointCloudData:
        filename = self.filenames[index]

        (points_action, points_anchor) = self.load_data(
            filename,
            action_class=self.action_class,
            anchor_class=self.anchor_class,
        )

        if self.distractor_anchor_aug:
            (
                _,
                points_action,
                points_anchor1,
                points_anchor2,
                debug,
            ) = get_random_distractor_demo(
                None,
                torch.from_numpy(points_action),
                torch.from_numpy(points_anchor),
                transform_base=self.multimodal_transform_base,
                rot_sample_method=self.distractor_rot_sample_method,
            )
            points_action = points_action.numpy()
            points_anchor = torch.cat([points_anchor1, points_anchor2], dim=1).numpy()

        # Apply occlusions
        if self.occlusion_cfg is not None:
            points_action = self.occlusion_fn(
                points_action, self.action_class, self.min_num_points
            )
            points_anchor = self.occlusion_fn(
                points_anchor, self.anchor_class, self.min_num_points
            )

        # Downsample
        points_action, _ = maybe_downsample(points_action, self.min_num_points)
        points_anchor, _ = maybe_downsample(points_anchor, self.min_num_points)
        np.random.shuffle(points_action)
        np.random.shuffle(points_anchor)

        action_symmetry_features = None
        anchor_symmetry_features = None
        action_symmetry_rgb = None
        anchor_symmetry_rgb = None

        # # Symmetry
        # (
        #     action_symmetry_features,
        #     anchor_symmetry_features,
        #     action_symmetry_rgb,
        #     anchor_symmetry_rgb,
        # ) = new_compute_demo_symmetry_features(
        #     points_action[0],
        #     points_anchor[0],
        #     OBJECT_LABELS_TO_CLASS[(self.object_type, self.action_class)],  # type: ignore
        #     OBJECT_LABELS_TO_CLASS[(self.object_type, self.anchor_class)],  # type: ignore
        # )

        # assert not isinstance(action_symmetry_features, torch.Tensor)
        # assert not isinstance(anchor_symmetry_features, torch.Tensor)

        # if action_symmetry_features is not None:
        #     action_symmetry_features = np.expand_dims(
        #         action_symmetry_features, 0
        #     ).astype(np.float32)
        #     anchor_symmetry_features = np.expand_dims(
        #         anchor_symmetry_features, 0
        #     ).astype(np.float32)
        #     action_symmetry_rgb = np.expand_dims(action_symmetry_rgb, 0)
        #     anchor_symmetry_rgb = np.expand_dims(anchor_symmetry_rgb, 0)

        return {
            "points_action": points_action,
            "points_anchor": points_anchor,
            "rgb_action": None,
            "rgb_anchor": None,
            "action_symmetry_features": action_symmetry_features,
            "anchor_symmetry_features": anchor_symmetry_features,
            "action_symmetry_rgb": action_symmetry_rgb,
            "anchor_symmetry_rgb": anchor_symmetry_rgb,
            "phase": None,
            "phase_onehot": None,
        }

    def __len__(self) -> int:
        return len(self.filenames)


@dataclass
class CustomPretrainingPointCloudDatasetConfig:
    dataset_root: str
    data_fn: str = "{}_valid_pcs_list.txt"
    pc_aug: bool = False
    phase: str = "train"
    obj_class: str = "all"
    num_points: int = 1024
    shuffle_pcs: bool = True
    dataset_type: str = "depth_pretraining"
    cloud_type: str = "final"
    data_size: int = -1


class CustomPretrainingPointCloudDataset(Dataset):
    def __init__(self, cfg: CustomPretrainingPointCloudDatasetConfig, overfit=-1):
        # 保留核心参数
        self.data_dir = cfg.dataset_root
        self.num_points = cfg.num_points
        self.pc_aug = cfg.pc_aug
        # 读取预处理后的样本列表
        self.data_list = self._read_data(cfg.data_fn.format(cfg.phase))

        # 过滤/裁剪样本（兼容原有逻辑）
        if overfit > 0:
            self.data_list = self.data_list[:overfit]
        if cfg.shuffle_pcs:
            random.shuffle(self.data_list)
        print(f"Dataset length (final): {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def _read_data(self, data_fn):
        """读取预处理后的有效样本列表"""
        # 缓存逻辑（兼容原有）
        pre_compute_file_name = f"custom_pcs_meta_list_{osp.basename(data_fn)}"
        cache_path = osp.join(self.data_dir, pre_compute_file_name)
        
        # 加载缓存
        if osp.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # 读取样本列表文件
        with open(osp.join(self.data_dir, data_fn), 'r') as f:
            data_list = [line.strip() for line in f.readlines()]
        
        # 过滤存在的文件
        data_list = [p for p in data_list if osp.exists(p) and is_valid_file(p)]
        
        # 缓存列表
        with open(cache_path, 'wb') as f:
            pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        return data_list

    # def _is_valid_file(self, filename: str, pc_key: str) -> bool:
    #     """检查 npz 文件中的点云数据是否有效（无 inf/nan 且数值范围合理）"""
    #     try:
    #         # 只加载 'action' 数组，不进行任何预处理
    #         with np.load(filename, allow_pickle=True) as point_data:
    #             points_action_np = point_data[pc_key]
    #             mean_ = points_action_np.mean(axis=0)
    #             # 1. 检查是否包含 inf 或 nan
    #             if np.isinf(points_action_np).any() or np.isinf(mean_).any():
    #                 return False
                
    #             # 2. 检查数值范围是否过大（阈值 1e5 可根据实际数据调整）
    #             #    避免后续计算 mean 时溢出（float32 范围约 3e38，但极大会导致精度问题）
    #             if np.abs(points_action_np).max() > 1e5:
    #                 return False
                
    #             if np.isnan(points_action_np).any() or np.isnan(mean_).any():
    #                 return False

    #             return True
    #     except Exception as e:
    #         print(f"Error loading or checking file {filename}: {e}")
    #         return False

    @staticmethod
    def _recenter_pc(pc):
        """点云中心化（复用原有逻辑）"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid[None]
        return pc, centroid

    @staticmethod
    def _shuffle_pc(pc):
        """打乱点云（复用原有逻辑）"""
        order = np.arange(pc.shape[0])
        random.shuffle(order)
        return pc[order]

    def _sample_fixed_points(self, part_pcs):
        """对每个零件点云采样固定总数的点
        args:
            part_pcs: 零件点云列表 [P1, P2, ..., Pn]
        return:
            total_pcs: sample points [P1, P2, ..., Pn], here len(Pi) is fixed
            nps: 各零件采样点数
        """
        total_pcs = []
        nps = []  # 各零件采样点数
        num_parts = len(part_pcs)
        
        # 按零件数均分点数，最后一个零件补全
        base_n = self.num_points // num_parts
        remain_n = self.num_points % num_parts
        
        for i, pc in enumerate(part_pcs):
            n = base_n + (1 if i == num_parts - 1 else 0) * remain_n
            # 采样/填充到指定点数
            if len(pc) >= n:
                sample_idx = np.random.choice(len(pc), n, replace=False)
                sampled_pc = pc[sample_idx]
            else:
                sample_idx = np.random.choice(len(pc), n, replace=True)
                sampled_pc = pc[sample_idx]
            total_pcs.append(sampled_pc)
            nps.append(n)

        return total_pcs, nps

    def _point_clound_augmentation(self, pcs) -> list:
        if not isinstance(pcs, list):
            pcs = [pcs]
        for i, pc in enumerate(pcs):
            noise = np.random.uniform(-0.005, 0.005, pc.shape)
            # TODO: maybe added noise * bounding box size ?
            pcs[i] += noise
        return pcs

    @functools.lru_cache(maxsize=100)
    def load_data(self, filename) -> list:
        with open(filename, 'rb') as f:
            sample_data = pickle.load(f)
        return sample_data['part_pcs']

    def __getitem__(self, index):
        """重写样本读取逻辑"""
        # 1. 加载预处理后的点云和元信息
        sample_path = self.data_list[index]
        part_pcs = self.load_data(sample_path)
        # 2. 对点云采样固定总数的点
        cur_pts, nps = self._sample_fixed_points(part_pcs)
        if self.pc_aug:
            cur_pts = self._point_clound_augmentation(cur_pts)
        for i, pc in enumerate(cur_pts):
            # 3. 洗牌（兼容原有逻辑）
            # rc_pts, _ = self._recenter_pc(pc)  # NOTE: 中心化在网络前向传播中完成
            cur_pts[i] = self._shuffle_pc(pc)

        cur_pts = np.concatenate(cur_pts, axis=0).astype(np.float32)

        # 6. 构造返回字典（完全兼容原有格式）
        data_dict = {
            "part_pcs": torch.from_numpy(cur_pts),  # [N_sum, 3]
            "n_pcs": torch.tensor(nps, dtype=torch.long),  # [max_num_part]
            "data_id": index,
        }
        return cur_pts


class CustomPretrainingTotalPCDataset(CustomPretrainingPointCloudDataset):
    def __init__(self, cfg: CustomPretrainingPointCloudDatasetConfig, overfit=-1):
        assert cfg.dataset_type == "total_pcs_pretraining", \
            "Only support total_pcs_pretraining"
        self.data_size = cfg.data_size
        super().__init__(cfg, overfit)

    def _read_data(self, data_fn):
        filenames = fnmatch.filter(os.listdir(self.data_dir), f"**_asm_**.npz")
        filenames = [
            os.path.join(self.data_dir, fn) for fn in filenames
        ]
        original_length = len(filenames)

        bad_demo_names = []
        for i in range(len(filenames)):
            filename = filenames[i]
            if i == 0:
                print(filename)
            if not os.path.exists(filename):
                bad_demo_names.append(filename)
                continue
            if not is_valid_file(filename):
                bad_demo_names.append(filename)
                continue

        # remove bad filenames in bad_demo
        if len(bad_demo_names) > 0:
            filenames = [name for name in filenames if name not in bad_demo_names]
        
        print(f"Total valid files: {len(filenames)} / {original_length}")
        assert len(filenames) > 0, "No valid files found at %s" % self.data_dir
        if self.data_size > 0 and len(filenames) < self.data_size:
            idx = np.random.choice(len(filenames), self.data_size, replace=True)
            filenames = [filenames[i] for i in idx]

        return filenames

    @functools.lru_cache(maxsize=100)
    def load_data(self, filename) -> list:
        point_data = np.load(filename, allow_pickle=True)
        if random.random() > 0.5:
            points_action_np = point_data['anchor'].copy()
        else:
            points_action_np = point_data['action'].copy()
        points_action_mean_np = points_action_np.mean(axis=0)
        points_action_np = points_action_np - points_action_mean_np

        points_action = points_action_np.astype(np.float32)[None, ...]

        return list(points_action)