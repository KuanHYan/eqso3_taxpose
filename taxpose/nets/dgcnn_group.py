import torch
from torch import nn
from pointnet2_ops import pointnet2_utils
from taxpose.nets.point_net_util import get_graph_feature
# knn = KNN(k=16, transpose_mode=False)


class DGCNN_Grouper(nn.Module):
    def __init__(self, emb_dims=256, output_num=256, knn=16, dropout=0.1, norm='BN'):
        super().__init__()
        '''
        K has to be 16
        '''
        self.k = knn
        self.emb_dims = emb_dims
        self.norm = norm
        if norm == 'BN':
            norm1 = nn.BatchNorm2d(64)
            norm2 = nn.BatchNorm2d(256)
            norm3 = nn.BatchNorm2d(1024)
            # norm4 = nn.BatchNorm2d(emb_dims)
        elif norm == 'GN':
            norm1 = nn.GroupNorm(4, 64)
            norm2 = nn.GroupNorm(4, 256)
            norm3 = nn.GroupNorm(4, 1024)
            # norm4 = nn.GroupNorm(4, emb_dims)
        else:
            raise ValueError('Invalid normalization: %s' % norm)

        self.input_trans = nn.Conv1d(3, 8, 1)
        # self.layer1 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=1, bias=False),
        #                             norm1,
        #                             nn.LeakyReLU(negative_slope=0.2))

        # self.layer2 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
        #                             norm2,
        #                             nn.LeakyReLU(negative_slope=0.2))

        # self.layer3 = nn.Sequential(nn.Conv2d(128*2, 128, kernel_size=1, bias=False),
        #                             norm3,
        #                             nn.LeakyReLU(negative_slope=0.2))

        # self.layer4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
        #                             norm4,
        #                             nn.LeakyReLU(negative_slope=0.2))
        
        self.layer1 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=1, bias=False),
                                    norm1,
                                    nn.LeakyReLU(negative_slope=0.2))

        self.layer2 = nn.Sequential(nn.Conv2d(72*2, 256, kernel_size=1, bias=False),
                                    norm2,
                                    nn.LeakyReLU(negative_slope=0.2))

        self.layer3 = nn.Sequential(nn.Conv2d(256*2, 1024, kernel_size=1, bias=False),
                                    norm3,
                                    nn.LeakyReLU(negative_slope=0.2))
        fpn_dim = 8 + 64 + 256 + 1024
        # self.layer4 = nn.Sequential(nn.Conv1d(fpn_dim, emb_dims, kernel_size=1, bias=False),
        #                             norm4,
        #                             nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(fpn_dim, 512, kernel_size=1, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.output_num = output_num

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def forward(self, x, down=True):

        # x: bs, 3, np
        coor = x
        f1 = self.input_trans(x)

        f = get_graph_feature(coor, f1, coor, f1, self.k)
        f = self.dropout(self.layer1(f))
        f2 = f.max(dim=-1, keepdim=False)[0]  # bs, 64, np//2, k -> bs, 64, np//2

        f_ = torch.cat([f1, f2], dim=1)

        if down:
            coor_q, f_q = self.fps_downsample(coor, f_, self.output_num)
        else:
            coor_q, f_q = coor, f_
        
        f = get_graph_feature(coor_q, f_q, coor, f_, self.k)
        f = self.dropout(self.layer2(f))
        f3 = f.max(dim=-1, keepdim=False)[0]  # bs, 128, onp, k -> bs, 64, onp
        coor = coor_q

        f = get_graph_feature(coor, f3, coor, f3, self.k)
        f = self.dropout(self.layer3(f))
        f4 = f.max(dim=-1, keepdim=False)[0]  # bs, 128, onp, k -> bs, 128, onp

        # coor_q, f_q = self.fps_downsample(coor, f, self.output_num)
        # f = self.get_graph_feature(coor_q, f_q, coor, f)
        # f = self.layer4(f)
        # f = self.dropout(f.max(dim=-1, keepdim=False)[0])
        # coor = coor_q

        f = torch.cat([f_q, f3, f4], dim=1)
        f = self.conv5(f)
        if not down:
            return f
        return (f, coor)



if __name__ == '__main__':
    x = torch.rand(2, 3, 1024).cuda()
    grouper = DGCNN_Grouper(emb_dims=512, output_num=512).cuda()
    coor, f = grouper(x)
    print(coor.shape)
    print(f.shape)
