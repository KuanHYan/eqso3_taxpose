import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from taxpose.nets.dgcnn_gc import DGCNN_GC
from taxpose.nets.pointnet import PointNet
from taxpose.utils.se3 import dualflow2pose
from third_party.dcp.model import (
    PositionwiseFeedForward,
    SublayerConnection,
    clones,
)
from taxpose.nets.huggingface_tf import (
    LinearMHAttnWithKNN,
    MHAttention,
    LinearMHAttention,
    MHAttnWithKNN
)


def knn_index_to_mask(knn_index, num_points, device):
    """将 KNN 索引 (B, N, K) 转换为注意力掩码 (B, N, N).
    邻居位置为 0，非邻居位置为 -inf.
    """
    B, N, K = knn_index.shape
    mask = torch.full((B, N, N), float('-inf'), device=device)
    batch_idx = torch.arange(B, device=device).view(-1, 1, 1).expand(B, N, K)
    point_idx = torch.arange(N, device=device).view(1, -1, 1).expand(B, N, K)
    mask[batch_idx, point_idx, knn_index] = 0.0
    return mask


class EncoderLayer(nn.Module):
    """支持 knn_index 的 EncoderLayer。

    若传入 knn_index，自注意力将仅关注 K 近邻点；
    否则为全注意力。
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None, knn_index=None):
        # 若提供了 knn_index，构造局部注意力掩码
        local_mask = mask
        # if knn_index is not None:
        #     B, N, _ = knn_index.shape
        #     knn_mask = knn_index_to_mask(knn_index, N, x.device)
        #     local_mask = knn_mask if mask is None else mask + knn_mask

        x = self.sublayer[0](x,
                    lambda x: self.self_attn(x, x, x, mask=local_mask, knn_index=knn_index))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """支持 knn_index 的 Encoder。"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask=None, knn_index=None):
        for layer in self.layers:
            x = layer(x, mask, knn_index)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """支持 knn_index 的 DecoderLayer。

    自注意力可使用 knn_index 做局部约束；
    src-attn 保持全注意力（跨点云匹配不应限制邻居）。
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask=None, tgt_mask=None, knn_index=None):
        # self-attn: 可使用 knn_index 限制邻域
        lc_tgt_mask = tgt_mask
        # if knn_index is not None:
        #     B, N, _ = knn_index.shape
        #     knn_mask = knn_index_to_mask(knn_index, N, x.device)
        #     lc_tgt_mask = knn_mask if tgt_mask is None else tgt_mask + knn_mask

        m = memory
        x = self.sublayer[0](x,
                    lambda x: self.self_attn(x, x, x, mask=lc_tgt_mask, knn_index=knn_index))
        # src-attn: 全注意力（跨点云匹配）
        x = self.sublayer[1](x,
                             lambda x: self.src_attn(x, m, m, mask=src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """支持 knn_index 的 Decoder。"""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask=None, tgt_mask=None, knn_index=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, knn_index)
        return self.norm(x)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, knn_index=None):
        "Take in and process masked src and target sequences."
        return self.decode(
            self.encode(src, src_mask, knn_index=knn_index), src_mask,
            tgt, tgt_mask,
            knn_index=knn_index
        )

    def encode(self, src, src_mask, knn_index=None):
        return self.encoder(self.src_embed(src), src_mask, knn_index=knn_index)

    def decode(self, memory, src_mask, tgt, tgt_mask, knn_index=None):
        return self.generator(
            self.decoder(
                self.tgt_embed(tgt), memory,
                src_mask, tgt_mask,
                knn_index=knn_index
            ))


class CustomTransformer(nn.Module):
    """This is a custom transformer model that is used to embed the point clouds.
    It is based on the transformer model from the DCP paper.

    See: https://github.com/WangYueFt/dcp/blob/master/model.py
    """

    def __init__(
        self,
        emb_dims=512,
        n_blocks=1,
        dropout=0.0,
        ff_dims=1024,
        n_heads=4,
        return_attn=False,
        bidirectional=True,
        attn_mode="torch_attn",
        **kwargs
    ):
        super(CustomTransformer, self).__init__()
        self.emb_dims = emb_dims
        self.N = n_blocks
        self.dropout = dropout
        self.ff_dims = ff_dims
        self.n_heads = n_heads
        self.return_attn = return_attn
        self.bidirectional = bidirectional
        self.attn_mode = attn_mode
        c = copy.deepcopy
        if self.attn_mode == "linear":
            attn_class = LinearMHAttention
        elif self.attn_mode == "linear_with_knn":
            attn_class = LinearMHAttnWithKNN
        elif self.attn_mode == "torch_attn_with_knn":
            attn_class = MHAttnWithKNN
        else:
            attn_class = MHAttention

        attn = attn_class(
            self.n_heads,
            self.emb_dims,
            dropout=self.dropout,
            project_bias=False,
            need_weights=self.return_attn
        )
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
            Decoder(
                DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout),
                self.N,
            ),
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(),
        )

    def get_attn_scores(self, query_emb, tgt_emb, seq_dim=1, knn_index=None):
        assert self.return_attn, "return_attn must be True to get attention scores"
        if seq_dim == 2:
            query_emb = query_emb.transpose(2, 1).contiguous()  # (batch, seq, channels)
            tgt_emb = tgt_emb.transpose(2, 1).contiguous()

        emb = self.model.forward(
            query_emb, tgt_emb, None, None,
            knn_index=knn_index).transpose(2, 1).contiguous()
        src_attn = self.model.decoder.layers[-1].src_attn.attn
        return src_attn.mean(dim=1), emb  # B, H, N, M --> B, N, M

    def forward(self, *input, knn_index=None):
        query = input[0]
        key = input[1]

        src_embedding = self.model.forward(
            key, query, None, None,
            knn_index=knn_index).transpose(2, 1).contiguous()

        src_attn = self.model.decoder.layers[-1].src_attn.attn if self.return_attn else None

        outputs = {"src_embedding": src_embedding, "src_attn": src_attn}

        if self.bidirectional:
            tgt_embedding = (
                self.model(query, key, None, None, knn_index=knn_index).transpose(2, 1).contiguous()
            )
            tgt_attn = self.model.decoder.layers[-1].src_attn.attn if self.return_attn else None

            outputs = {
                **outputs,
                "tgt_embedding": tgt_embedding,
                "tgt_attn": tgt_attn,
            }

        return outputs


