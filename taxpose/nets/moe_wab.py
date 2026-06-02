import torch
from torch import nn
from taxpose.nets.vn_dgcnn import VN4Head
from torch.nn.utils.weight_norm import weight_norm


class ExpertNetwork(nn.Module):
	def __init__(self, hidden_size, intermediate_size):
		super().__init__()
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size

		self.linear1 = nn.Linear(hidden_size, intermediate_size)
		self.linear2 = nn.Linear(intermediate_size, hidden_size)

	def forward(self, x):
		x = self.linear1(x)
		x = nn.functional.relu(x)
		output = self.linear2(x)
		return output
	

class VN4ExpertNetwork(nn.Module):
	def __init__(self, pts_num):
		super().__init__()
		self.net = VN4Head(pts_num)

	def forward(self, x):
		x = x.unsqueeze(0)
		x = self.net(x)
		x = x.squeeze(0)
		return x

class MLPExpertNetwork(nn.Module):
	def __init__(self, pts_num):
		super().__init__()
		self.net = nn.Linear(pts_num, pts_num, bias=False)
	def forward(self, x):
		x = x.unsqueeze(0)
		x = self.net(x)
		x = x.squeeze(0)
		return x

class Router(nn.Module):
	def __init__(self, hidden_size, expert_num, top_k):
		super().__init__()
		self.router = nn.Linear(hidden_size, expert_num)
		self.top_k = top_k
		self.hidden_size = hidden_size

	def forward(self, x):
		x = x.view(-1, self.hidden_size)
		x = self.router(x)
		x = nn.functional.softmax(x, dim=-1)
		topk_weight, topk_idx = torch.topk(x, k=self.top_k, dim=-1, sorted=False)
		# 对topk权重重新归一化
		topk_weight = topk_weight/topk_weight.sum(dim=-1, keepdim=True)
		return topk_weight, topk_idx


class MOELayer(nn.Module):
    def __init__(self, fea_dim, point_num, expert_num, top_k):
        super().__init__()
        self.experts = nn.ModuleList([MLPExpertNetwork(point_num) for _ in range(expert_num)])
        self.share_expert = MLPExpertNetwork(point_num)
        self.router = Router(fea_dim, expert_num, top_k) # 路由器
        self.fea_pool = nn.AdaptiveAvgPool1d(1)  # d, N_pts --> d, 1
        self.top_k = top_k

    def forward(self, x):
        """Args
            x: points with shape of (B, 3, N)
            fea: features with shape of (B, C, N)
        """
        x, fea = x[0], x[1]
        batch_size, _, pts_n = x.size()
        pool_fea = self.fea_pool(fea) # 展平：(B, hidden_size)
        # 路由器为每个token选择top-k个专家及对应权重
        topk_weight, topk_idx = self.router(pool_fea) # 形状均为 (B, top_k)
        output = self.share_expert(x)
        # 对每个token，累加其top-k专家的加权输出
        for i in range(batch_size):
            for j in range(self.top_k):
                expert = self.experts[topk_idx[i, j]]
                output[i] += topk_weight[i, j] * expert(x[i])

        return output.view(batch_size, -1, pts_n)
	

if __name__ == "__main__":
	pts = torch.rand(2, 3, 256)
	fea = torch.rand(2, 512, 256)
	net = MOELayer(512, 256, 16, top_k=1)
	out = net(pts, fea)
	print(out.shape)
