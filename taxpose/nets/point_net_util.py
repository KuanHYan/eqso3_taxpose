import torch
from pytorch3d.ops import sample_farthest_points


def fps_downsample(coor, x, num_group):
    """FPS downsample using pytorch3d.

    coor: (B, 3, N)  -- coordinates
    x:    (B, C, N)  -- features
    num_group: int   -- number of output points K
    Returns: new_coor (B, 3, K), new_x (B, C, K)
    """
    xyz = coor.transpose(1, 2).contiguous()                # (B, N, 3)
    _, fps_idx = sample_farthest_points(xyz, K=num_group)  # (B, K)

    # Gather on combined (B, C_total, N) using expanded indices
    combined_x = torch.cat([coor, x], dim=1)               # (B, 3+C, N)
    idx_expanded = fps_idx[:, None, :].expand(-1, combined_x.size(1), -1)  # (B, 3+C, K)
    new_combined_x = combined_x.gather(dim=2, index=idx_expanded)  # (B, 3+C, K)

    new_coor = new_combined_x[:, :3]   # (B, 3, K)
    new_x = new_combined_x[:, 3:]      # (B, C, K)
    return new_coor, new_x


def knn(k, query, key):
    """
    计算 query (B,N,C) 与 key (B,M,C) 之间的 k 近邻.
        Input:
            k: max sample number in local region
            xyz: all points, [B, N, C]
            new_xyz: query points, [B, M, C]
        Return:
            dist: squared distance, [B, N, k]
            idx: grouped points index, [B, N, k]
    """
    dist = torch.cdist(query, key)  # (B, N, M)
    dist, idx = torch.topk(dist, k, dim=-1, largest=False)      # (B, N, k)
    return dist, idx


def get_graph_feature(x_q, x_k=None, k=20, coord_q=None, coord_k=None):
    """
    coord: bs, 3, np; x: bs, c, np
    if coord_k is None, use coord_k as coord_k
    if x is None, use x as coord
    """
    batch_size = x_q.size(0)
    num_points_q = x_q.size(2)
    x_q = x_q.transpose(2, 1).contiguous()  # B, N, C
    if x_k is None:
        x_k = x_q
    else:
        x_k = x_k.transpose(2, 1).contiguous()
    num_points_k = x_k.size(1)
    num_dims = x_k.size(2)
    if coord_q is None:
        coord_q = x_q
    else:
        coord_q = coord_q.transpose(2, 1).contiguous()
    if coord_k is None:
        coord_k = coord_q
    else:
        coord_k = coord_k.transpose(2, 1).contiguous()

    _, idx = knn(k, coord_q, coord_k)
    idx = idx.transpose(-1, -2).contiguous()  # B K N
    assert idx.shape[1] == k
    idx_base = torch.arange(0, batch_size, device=coord_q.device).view(-1, 1, 1) * num_points_k
    idx = idx + idx_base
    idx = idx.view(-1)
    
    feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
    feature = feature.view(batch_size, k, num_points_q, num_dims)
    x_q = x_q.view(batch_size, 1, num_points_q, num_dims).expand(-1, k, -1, -1)
    feature = torch.cat((feature - x_q, x_q), dim=-1)
    return feature.permute(0, 3, 2, 1).contiguous()


def get_graph_feature_for_vndgcnn(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(-1)
    x = x.view(batch_size, -1, num_points)
    x = x.transpose(2, 1).contiguous()  # (batch_size, N, C)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            _, idx = knn(k, x, x)
        else:          # fixed knn graph with input point coordinates
            if x_coord.size(1) == 3:
                x_coord = x_coord.transpose(2, 1).contiguous()
            _, idx = knn(k, x_coord, x_coord)
    assert idx.shape[-1] == k

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    num_dims = x.size(-1)
    num_dims = num_dims // 3  # for vn

    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature


if __name__ == '__main__':
    x_q = torch.rand(2, 8, 100)
    x_k = torch.rand(2, 8, 50)
    # coord_q = torch.rand(2, 3, 100)
    # coord_k = torch.rand(2, 3, 50)
    # print(knn(20, coord_q.transpose(2, 1), coord_k.transpose(2, 1))[1].shape)
    print(get_graph_feature(x_q).shape)