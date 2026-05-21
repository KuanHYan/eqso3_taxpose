import torch
import torch.nn as nn
import math
import open3d as o3d
import open3d.core as o3c


def calc_ppf_gpu(points, point_normals, patches, patch_normals):
    '''
    Calculate ppf gpu
    points: [b, n, 3]
    point_normals: [b, n, 3]
    patches: [b, n, nsamples, 3]
    patch_normals: [b, n, nsamples, 3]
    '''
    points = torch.unsqueeze(points, dim=2).expand(-1, -1, patches.shape[2], -1)
    point_normals = torch.unsqueeze(point_normals, dim=2).expand(-1, -1, patches.shape[2], -1)
    vec_d = patches - points  # [b, n, n_samples, 3]
    d = torch.sqrt(torch.sum(vec_d ** 2, dim=-1, keepdim=True))  # [b, n, n_samples, 1]
    # angle(n1, vec_d)
    y = torch.sum(point_normals * vec_d, dim=-1, keepdim=True)
    x = torch.cross(point_normals, vec_d, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle1 = torch.atan2(x, y) / math.pi

    # angle(n2, vec_d)
    y = torch.sum(patch_normals * vec_d, dim=-1, keepdim=True)
    x = torch.cross(patch_normals, vec_d, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle2 = torch.atan2(x, y) / math.pi

    # angle(n1, n2)
    y = torch.sum(point_normals * patch_normals, dim=-1, keepdim=True)
    x = torch.cross(point_normals, patch_normals, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle3 = torch.atan2(x, y) / math.pi

    ppf = torch.cat([d, angle1, angle2, angle3], dim=-1) #[b, n, samples, 4]
    return ppf.mean(dim = -2, keepdim = True)


def hierachical_knn_query_list(points, ref_points, k_list ):
    b, p, _ = points.shape
    _, r, _ = ref_points.shape
    dist = torch.cdist(points, ref_points)
    k = k_list[0]
    ret = [torch.topk(-dist, k)[1]]
    for i in range(1, len(k_list)):
        k = k_list[i]
        last_k = k_list[i-1]

        knn_index = torch.topk(-dist, k)[1][:, :, last_k:]
        ret.append(knn_index)
        # ret.append(ref_points[torch.arange(b)[:,None,None], knn_index])
    return ret


class ManualPointWiseGemoFea(nn.Module):
    def __init__(self, project, embedding_dim=512, sample_num=300):
        super(ManualPointWiseGemoFea, self).__init__()
        self.ppf_nn = [10, 20, 40, 80, 160, 300]
        self.camera_cfg = o3c.Tensor([0.0, 0.0, 0.0], device=o3c.Device("CUDA:0"), dtype=o3c.Dtype.Float32)
        if project:
            self.pos_project = nn.Linear(len(self.ppf_nn)*4, embedding_dim, bias=False).cuda()
        self.project = project

    def cal_normal(self, points, rand_rotaton=None, translation=None, size=None):
        b, n, _ = points.shape
        # translation = translation/size.norm(dim=-1, keepdim=True)
        # points = self.rotate_pts_batch(points, rand_rotaton) + \
        #     translation[:, None, :].repeat(1, n, 1)
        if points.device == torch.device('cpu'):  # 确保张量在正确的设备上
            self.camera_cfg = self.camera_cfg.to(o3c.Device("CPU:0"))
        rets = []
        for i in range(b):
            point = points[i]  # (N, 3) torch.Tensor on CUDA

            # 将 PyTorch CUDA 张量转换为 Open3D CUDA 张量（零拷贝）
            # 需要张量连续且为 float32
            if not point.is_contiguous():
                point = point.contiguous()
            # 使用 __dlpack__ 协议实现零拷贝共享 GPU 内存
            o3d_point = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(point))

            # 创建 GPU 点云对象
            pcd = o3d.t.geometry.PointCloud(o3d_point)
            # 估计法向量，参数与原 CPU 版本一致：radius=4.0, max_nn=10
            pcd.estimate_normals(max_nn=10, radius=4.0)
            # 统一法向量朝向（默认朝向相机位置 (0,0,0)）
            pcd.orient_normals_towards_camera_location(self.camera_cfg)

            # 取出法向量，转换回 PyTorch CUDA 张量
            normals_o3d = pcd.point.normals  # o3c.Tensor on CUDA
            # 同样使用 dlpack 零拷贝回 PyTorch
            ret = torch.utils.dlpack.from_dlpack(normals_o3d.to_dlpack())
            rets.append(ret)

        rets = torch.stack(rets, dim=0)  # (B, N, 3)
        # rets = self.rotate_pts_batch(rets, rand_rotaton.transpose(1, 2))
        return rets

    def calc_ppf(self, pts, normal):
        b, n, _ = pts.shape
        k_list = self.ppf_nn
        knn_index_list = hierachical_knn_query_list(pts, pts, k_list)
        patches_list = []
        patches_normals_list = []
        for k, knn_index in zip(k_list, knn_index_list):
            # if k>1:
            #     knn_index = knn_index[:,:,1:]
            patches_list.append(
                pts[torch.arange(b)[:, None, None], knn_index])
            patches_normals_list.append(
                normal[torch.arange(b)[:, None, None], knn_index])
        ppfs = []
        for patches, patch_normals in zip(patches_list, patches_normals_list):
            ppfs.append(calc_ppf_gpu(pts, normal, patches, patch_normals).reshape(b, n, -1))
        
        return torch.cat(ppfs, dim = -1)

    def rotate_pts_batch(self, pts, rotation):
        pts_shape = pts.shape
        b = pts_shape[0]
        res = rotation[:, None, :, :] @ pts.reshape(b, -1, 3)[:, :, :, None]
        return res.squeeze().reshape(pts_shape)

    def forward(self, pts):
        """
        input: 
            pts with shape of [b, n, 3] or [b, 3, n]
            downsample_num: int
        return: 
            ppf_feature with shape of [b, c, n], default c=24
        """
        if pts.shape[1] == 3:
            pts = pts.permute(0, 2, 1)
        normals = self.cal_normal(pts)
        ppf_feature = self.calc_ppf(pts, normals)
        if self.project:
            ppf_feature = self.pos_project(ppf_feature)

        return ppf_feature.permute(0, 2, 1)


if __name__ == "__main__":
    import time
    device = torch.device('cuda:0')
    model = ManualPointWiseGemoFea(project=True)
    pts = torch.rand(4, 3, 1024).to(device)
    pts2 = torch.rand(4, 3, 512).to(device)
    pts3 = torch.rand(4, 3, 300).to(device)
    t0 = time.time()
    ppf_feature = model(pts)
    t1 = time.time()
    print(t1-t0)
    print(ppf_feature.shape)
    ppf_feature = model(pts2)
    t2 = time.time()
    print(t2-t1)
    ppf_feature = model(pts3)
    t3 = time.time()
    print(t3-t2)
