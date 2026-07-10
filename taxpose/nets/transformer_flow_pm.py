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
    MHAttention,
    MHAttnWithKNN,
    LinearMHAttention,
    LinearMHAttnWithKNN,
    PointAugmentedCrossAttention,
    PointAugmentedFFN,
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


class PointAugmentedDecoderLayer(nn.Module):
    """DecoderLayer that uses PointAugmentedCrossAttention for src-attn.

    Self-attn is unchanged (standard MHA).
    Src-attn uses PointAugmentedCrossAttention, which concatenates
    point coordinates to the value and returns both feature + weighted points.
    FFN uses PointAugmentedFFN to process features and points separately.

    Sub-layer structure:
        1. self-attn (standard residual + norm)
        2. src-attn (produces feat + corr_pts; residual only on feat)
        3. dual FFN  (processes feat + corr_pts independently)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn          # PointAugmentedCrossAttention
        self.feed_forward = feed_forward  # PointAugmentedFFN
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)
        self.norm3 = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None,
                knn_index=None, mem_pts=None):
        # ---- 1. self-attn (standard) ----
        x = x + self.dropout(
            self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                           mask=tgt_mask, knn_index=knn_index))

        # ---- 2. src-attn (point-augmented) ----
        src_out = self.src_attn(
            self.norm2(x), self.norm2(memory), self.norm2(memory),
            value_pts=mem_pts)
        if self.src_attn.return_attn:
            feat_src, corr_pts, _attn = src_out
        else:
            feat_src, corr_pts = src_out
        x = x + self.dropout(feat_src)
        # corr_pts has no residual (no previous corr_pts to add to)

        # ---- 3. dual FFN ----
        feat_ffn, corr_pts = self.feed_forward(
            self.norm3(x), corr_pts)
        x = x + self.dropout(feat_ffn)
        # corr_pts residual is applied inside PointAugmentedFFN

        return x, corr_pts


class PointAugmentedDecoder(nn.Module):
    """Stack of PointAugmentedDecoderLayers.

    Each layer independently uses the original anchor points as mem_pts.
    Only feature embeddings (x) propagate between layers; corr_pts from
    each layer are a separate output. The last layer's corr_pts are returned.
    """

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask=None, tgt_mask=None,
                knn_index=None, mem_pts=None):
        corr_pts = None
        for layer in self.layers:
            x, corr_pts = layer(x, memory, src_mask, tgt_mask,
                                knn_index, mem_pts)
        return self.norm(x), corr_pts


class EncoderDecoder(nn.Module):
    """A standard Encoder-Decoder architecture. Supports both standard and
    point-augmented decoders.

    When ``point_augmented=True``, the decoder is a ``PointAugmentedDecoder``
    and ``forward()`` accepts an optional ``mem_pts`` argument.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator,
                 point_augmented: bool = False):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.point_augmented = point_augmented

    def forward(self, src, tgt, src_mask, tgt_mask, knn_index=None,
                mem_pts=None):
        "Take in and process masked src and target sequences."
        return self.decode(
            self.encode(src, src_mask, knn_index=knn_index), src_mask,
            tgt, tgt_mask,
            knn_index=knn_index,
            mem_pts=mem_pts,
        )

    def encode(self, src, src_mask, knn_index=None):
        return self.encoder(self.src_embed(src), src_mask, knn_index=knn_index)

    def decode(self, memory, src_mask, tgt, tgt_mask, knn_index=None,
               mem_pts=None):
        if self.point_augmented:
            feat, corr_pts = self.decoder(
                self.tgt_embed(tgt), memory,
                src_mask, tgt_mask,
                knn_index=knn_index,
                mem_pts=mem_pts,
            )
            return self.generator(feat), corr_pts
        else:
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

        point_augmented = (attn_mode == "point_augmented")

        if self.attn_mode == "linear":
            attn_class = LinearMHAttention
        elif self.attn_mode == "linear_with_knn":
            attn_class = LinearMHAttnWithKNN
        elif self.attn_mode == "torch_attn_with_knn":
            attn_class = MHAttnWithKNN
        elif point_augmented:
            attn_class = MHAttention  # self-attn stays standard
        else:
            attn_class = MHAttention

        self_attn = attn_class(
            self.n_heads,
            self.emb_dims,
            dropout=self.dropout,
            project_bias=False,
            need_weights=False,
        )

        # ---- point-augmented specific modules ----
        if point_augmented:
            point_ffn_mode = kwargs.get("point_ffn_mode", "none")
            # Cross-attention with point augmentation
            src_attn = PointAugmentedCrossAttention(
                n_head=self.n_heads,
                d_model=self.emb_dims,
                dropout=self.dropout,
                project_bias=False,
                return_attn=self.return_attn,
            )
            # Dual FFN for decoder
            decoder_ff = PointAugmentedFFN(
                d_model=self.emb_dims,
                d_ff=self.ff_dims,
                num_points=kwargs.get("num_points", 1024),
                dropout=self.dropout,
                point_ffn_mode=point_ffn_mode,
                point_ffn_emb_dims=self.emb_dims,
            )
            # Standard FFN for encoder (unchanged)
            ff = PositionwiseFeedForward(
                self.emb_dims, self.ff_dims, self.dropout)
            decoder_layer = PointAugmentedDecoderLayer(
                self.emb_dims, c(self_attn), src_attn, decoder_ff, self.dropout,
            )
            decoder = PointAugmentedDecoder(decoder_layer, self.N)
        else:
            src_attn = attn_class(
                self.n_heads,
                self.emb_dims,
                dropout=self.dropout,
                project_bias=False,
                need_weights=self.return_attn,
            )
            ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
            decoder = Decoder(
                DecoderLayer(self.emb_dims, c(self_attn), src_attn, c(ff), self.dropout),
                self.N,
            )

        self.model = EncoderDecoder(
            Encoder(
                EncoderLayer(self.emb_dims, c(self_attn), c(ff), self.dropout), self.N),
            decoder,
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(),
            point_augmented=point_augmented,
        )

    def forward(self, *input, knn_index=None, mem_pts=None):
        query = input[0]
        key = input[1]

        if self.attn_mode == "point_augmented":
            result = self.model.forward(
                key, query, None, None,
                knn_index=knn_index, mem_pts=mem_pts,
            )
            src_embedding, action_corr_pts = result
            src_embedding = src_embedding.transpose(2, 1).contiguous()
            # action_corr_pts: (B, N, 3) already in correct format

            src_attn = None
            if self.return_attn:
                # Extract attention from last decoder layer
                last_layer = self.model.decoder.layers[-1]
                src_attn = last_layer.src_attn.attn
                if src_attn is not None:
                    src_attn = src_attn.mean(dim=1)  # (B, H, N, M) → (B, N, M)

            outputs = {
                "src_embedding": src_embedding,
                "src_attn": src_attn,
                "src_corr_pts": action_corr_pts,
            }
        else:
            src_embedding = self.model.forward(
                key, query, None, None,
                knn_index=knn_index).transpose(2, 1).contiguous()

            src_attn = self.model.decoder.layers[-1].src_attn.attn if self.return_attn else None
            if src_attn is not None:
                src_attn = src_attn.mean(dim=1)  # (B, H, N, M) → (B, N, M)

            outputs = {"src_embedding": src_embedding, "src_attn": src_attn}

        if self.bidirectional:
            if self.attn_mode == "point_augmented":
                result = self.model.forward(
                    query, key, None, None,
                    knn_index=knn_index, mem_pts=mem_pts,
                )
                tgt_embedding, anchor_corr_pts = result
                tgt_embedding = tgt_embedding.transpose(2, 1).contiguous()

                tgt_attn = None
                if self.return_attn:
                    last_layer = self.model.decoder.layers[-1]
                    tgt_attn = last_layer.src_attn.attn
                    if tgt_attn is not None:
                        tgt_attn = tgt_attn.mean(dim=1)

                outputs = {
                    **outputs,
                    "tgt_embedding": tgt_embedding,
                    "tgt_attn": tgt_attn,
                    "tgt_corr_pts": anchor_corr_pts,
                }
            else:
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


