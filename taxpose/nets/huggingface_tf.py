import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from torch.backends.cuda import sdp_kernel
# torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_math_sdp(False)


def get_graph_feature(feas, idx):
    """
    Args:
        feas: B, N, C
        idx: B, N, K
    Return:
        B, K, N, C
    """
    batch_size, num_points, num_dims = feas.size()
    k = idx.shape[-1]
    idx = idx.detach().transpose(-1, -2).contiguous()  # B K N

    idx_base = torch.arange(0, batch_size, device=feas.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    feature = feas.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, k, num_points, num_dims)
    feas = feas.view(batch_size, 1, num_points, num_dims).expand(-1, k, -1, -1)
    feature = torch.cat((feature - feas, feas), dim=-1)
    return feature


class Transformer(nn.Module):
    def __init__(
        self,
        emb_dims=512,
        n_blocks=1,
        dropout=0.0,
        ff_dims=1024,
        n_heads=4,
        return_attn=True,
        bidirectional=False
    ) -> None:
        super().__init__()
        
        self.emb_dims = emb_dims
        self.N = n_blocks
        self.dropout = dropout
        self.ff_dims = ff_dims
        self.n_heads = n_heads
        self.return_attn = return_attn
        self.bidirectional = bidirectional

        encoder_cfg = BertConfig(
            hidden_size=emb_dims, num_hidden_layers=n_blocks,
            num_attention_heads=n_heads, intermediate_size=ff_dims,
            hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
            max_position_embeddings=2048,
        )
        decoder_cfg = BertConfig(
            hidden_size=emb_dims, num_hidden_layers=n_blocks,
            num_attention_heads=n_heads, intermediate_size=ff_dims,
            hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
            is_decoder=True, add_cross_attention=True,
            max_position_embeddings=2048,
        )
        self.encoder = BertModel(encoder_cfg, add_pooling_layer=False)
        self.decoder = BertModel(decoder_cfg, add_pooling_layer=False)

    def _TFforward(self, query_emb, key_emb):
        encoder_outputs = self.encoder.forward(
            inputs_embeds=key_emb,
            output_hidden_states=True,
        )
        encoder_hidden_states = encoder_outputs[0]
        decoder_outputs = self.decoder.forward(
            encoder_hidden_states=encoder_hidden_states,
            inputs_embeds=query_emb,
            output_attentions=True,  # 开启注意力输出
            output_hidden_states=True
        )
        return decoder_outputs.cross_attentions[-1], decoder_outputs.last_hidden_state.transpose(2, 1).contiguous()

    def forward(self, *input):
        act_emb = input[0]  # (batch, channels, seq)
        tgt_emb = input[1]
        act_emb = act_emb.transpose(2, 1).contiguous()  # (batch, seq, channels)
        tgt_emb = tgt_emb.transpose(2, 1).contiguous()
        
        last_decoder_attn, out_embedding = self._TFforward(act_emb, tgt_emb)
        outputs = {"src_embedding": out_embedding, "src_attn": last_decoder_attn}

        if self.bidirectional:
            tgt_attn, tgt_embedding = self._TFforward(tgt_emb, act_emb)

            outputs = {
                **outputs,
                "tgt_embedding": tgt_embedding,
                "tgt_attn": tgt_attn,
            }

        return outputs
    
    def get_attn_scores(self, query_emb, tgt_emb, seq_dim=1):
        if seq_dim == 2:
            query_emb = query_emb.transpose(2, 1).contiguous()  # (batch, seq, channels)
            tgt_emb = tgt_emb.transpose(2, 1).contiguous()
        # query_emb = query_emb.view(query_emb.shape[0], -1, 1, query_emb.shape[-1])

        src_attn, _ = self._TFforward(query_emb, tgt_emb)
        return src_attn.mean(dim=1)  # B, H, N, M --> B, N, M


class MHAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1,
                 project_bias=False, need_weights=False):
        "Take in model size and number of heads."
        super(MHAttention, self).__init__()
        assert d_model % n_head == 0
        # We assume d_v always equals d_k
        self.attn_net = nn.MultiheadAttention(
            d_model, n_head,
            dropout=dropout,
            bias=project_bias,
            batch_first=True
        )
        self.need_weights = need_weights

    def forward(self, query, key, value, **kwargs):
        x, self.attn = self.attn_net.forward(
            query, key, value,
            need_weights=self.need_weights, average_attn_weights=False
        )
        return x


class MHAttnWithKNN(MHAttention):
    def __init__(self, n_head, d_model, dropout=0.1,
                 project_bias=False, need_weights=False):
        super().__init__(n_head, d_model, dropout,
                         project_bias, need_weights=need_weights)
        # PointTr: 
        self.knn_map = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.merge_map = nn.Linear(d_model*2, d_model)

    def forward(self, query, key, value, mask=None, knn_index=None):
        attn_fea = super().forward(query, key, value)
        if knn_index is not None:
            knn_f = get_graph_feature(query, knn_index)
            knn_f = self.knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            attn_fea = torch.cat([attn_fea, knn_f], dim=-1)
            attn_fea = self.merge_map(attn_fea)

        return attn_fea


# ──────────────────────────────────────────────────────────
#  Linear Attention (O(N d²) instead of O(N² d))
# ──────────────────────────────────────────────────────────

class LinearMHAttention(nn.Module):
    """Linear multi-head attention using kernel feature maps.

    Replaces softmax(QK^T/√d) V  with  φ(Q) · (φ(K)^T V) / φ(Q) · (φ(K)^T 1),
    reducing the computational complexity from O(N²d) to O(Nd²).

    Uses the kernel φ(x) = elu(x) + 1 (Katharopoulos et al., 2020).
    Compatible with the same calling convention as MHAttention /
    nn.MultiheadAttention.
    """

    def __init__(self, n_head: int, d_model: int, dropout: float = 0.1,
                 project_bias: bool = False, need_weights: bool = True):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_model = d_model
        self.eps = 1e-6

        self.W_q = nn.Linear(d_model, d_model, bias=project_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=project_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=project_bias)
        self.W_o = nn.Linear(d_model, d_model, bias=project_bias)
        self.dropout = nn.Dropout(dropout)
        self.attn = None          # stored attention weights (B,H,N,M) or (B,N,M)
        self.need_weights = need_weights

    @staticmethod
    def _kernel(x: torch.Tensor) -> torch.Tensor:
        """Kernel feature map: elu + 1  (non-negative, near-linear)."""
        return F.elu(x) + 1.0

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, **kwargs):
        """Forward pass.

        Args:
            query:  (B, N, d_model)
            key:    (B, M, d_model)
            value:  (B, M, d_model)
            mask:   unused (kept for API compatibility)

        Returns:
            output: (B, N, d_model)
        Side effect: stores self.attn (B, H, N, M) as kernel-based
            approximate attention weights.
        """
        B, N, _ = query.shape
        _, M, _ = key.shape
        H, d = self.n_head, self.d_k

        # ---- project & reshape → (B, H, *, d) ----
        Q = self.W_q(query).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        K = self.W_k(key).view(B, M, H, d).transpose(1, 2)
        V = self.W_v(value).view(B, M, H, d).transpose(1, 2)

        # ---- kernel feature map ----
        Q = self._kernel(Q)
        K = self._kernel(K)

        # ---- linear attention: O(N d²) ----
        # KV = Σ_j φ(k_j) ⊗ v_j          → (B, H, d, d)
        KV = torch.matmul(K.transpose(-2, -1), V)           # (B, H, d, d)
        # normalizer Z_i = Σ_j φ(q_i)·φ(k_j) → (B, H, N, 1)
        K_sum = K.sum(dim=-2, keepdim=True)                 # (B, H, 1, d)
        Z = 1.0 / (torch.matmul(Q, K_sum.transpose(-2, -1)) + self.eps)

        out = Z * torch.matmul(Q, KV)                       # (B, H, N, d)

        # ---- project back ----
        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.W_o(out)

        if self.need_weights:
            # ---- store kernel-based attention for downstream use ----
            # attn ≈ φ(Q) φ(K)^T  (already computed: Q, K are kernel features)
            attn = torch.matmul(Q, K.transpose(-2, -1))    # (B, H, N, M)
            self.attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return out


class LinearMHAttnWithKNN(LinearMHAttention):
    def __init__(self, n_head, d_model, dropout=0.1,
                 project_bias=False, need_weights=True):
        super().__init__(n_head, d_model, dropout, project_bias, need_weights)
        # PointTr: 
        self.knn_map = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.merge_map = nn.Linear(d_model*2, d_model)

    def forward(self, query, key, value, mask=None, knn_index=None):
        attn_fea = super().forward(query, key, value, mask)
        if knn_index is not None:
            knn_f = get_graph_feature(query, knn_index)
            knn_f = self.knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            attn_fea = torch.cat([attn_fea, knn_f], dim=-1)
            attn_fea = self.merge_map(attn_fea)

        return attn_fea


# class LinearCustomTransformer(CustomTransformer):
#     """CustomTransformer variant that uses linear attention throughout.

#     Identical interface to CustomTransformer but replaces every
#     nn.MultiheadAttention-based MHAttention with LinearMHAttention
#     (O(Nd²) instead of O(N²d)).

#     ``forward()`` and ``get_attn_scores()`` are inherited unchanged from
#     CustomTransformer — they only access ``self.model`` and
#     ``decoder.layers[-1].src_attn.attn``, which work identically with
#     LinearMHAttention (it also stores ``.attn`` as a side effect).
#     """

#     def __init__(
#         self,
#         emb_dims=512,
#         n_blocks=1,
#         dropout=0.0,
#         ff_dims=1024,
#         n_heads=4,
#         return_attn=False,
#         bidirectional=True,
#     ):
#         nn.Module.__init__(self)
#         self.emb_dims = emb_dims
#         self.N = n_blocks
#         self.dropout = dropout
#         self.ff_dims = ff_dims
#         self.n_heads = n_heads
#         self.return_attn = return_attn
#         self.bidirectional = bidirectional
#         c = copy.deepcopy

#         attn = LinearMHAttention(
#             self.n_heads, self.emb_dims,
#             dropout=self.dropout,
#             project_bias=False,
#         )
#         ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
#         self.model = EncoderDecoder(
#             Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
#             Decoder(
#                 DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout),
#                 self.N,
#             ),
#             nn.Sequential(),
#             nn.Sequential(),
#             nn.Sequential(),
#         )


# ──────────────────────────────────────────────────────────
#  Point-Augmented Cross Attention
#  将点云拼接到 Value 中，通过 scaled_dot_product_attention 一次性
#  完成 embedding 和点的加权求和，避免在 Head 中重复计算 corr_points
# ──────────────────────────────────────────────────────────

class PointAugmentedCrossAttention(nn.Module):
    """Cross-attention that concatenates point coordinates to the value.

    Instead of returning attention weights for the head to compute
    ``corr_points = anchor_points @ softmax(QK^T)``, this module directly
    produces weighted point coordinates alongside the attention output.

    This enables Flash Attention (via ``F.scaled_dot_product_attention``)
    since we no longer need to materialize the full N×M attention matrix.

    Shapes:
        query:      (B, N, d_model)
        key:        (B, M, d_model)
        value:      (B, M, d_model)
        value_pts:  (B, M, 3)

        feat_out:       (B, N, d_model)
        weighted_pts:   (B, N, 3)
        attn_weights:   (B, H, N, M)   if return_attn=True, else None
    """

    def __init__(self, n_head: int, d_model: int, dropout: float = 0.1,
                 project_bias: bool = False, return_attn: bool = False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.d_model = d_model
        self.return_attn = return_attn
        self.scale = self.d_k ** -0.5

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model, bias=project_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=project_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=project_bias)
        # output projection (only for feature part, not points)
        self.W_o = nn.Linear(d_model, d_model, bias=project_bias)
        self.dropout_attn = nn.Dropout(dropout)
        self.attn = None  # stored attn weights

        # Orthonormal projection to embed 3D points → d_k dims for Flash SDPA.
        # W_pts @ W_pts^T = I_3, so out = SDPA(Q, K, pts @ W_pts) @ W_pts^T
        # is mathematically equivalent to softmax(QK^T) @ pts.
        W = torch.randn(3, self.d_k)
        U, _, Vt = torch.linalg.svd(W, full_matrices=False)
        self.register_buffer('W_pts', (U @ Vt).to(torch.float32))  # (3, d_k)

    def _reshape_multihead(self, x: torch.Tensor, H: int) -> torch.Tensor:
        """(B, L, C) → (B, H, L, C//H)."""
        B, L, C = x.shape
        return x.view(B, L, H, C // H).transpose(1, 2)  # (B, H, L, d)

    def _merge_multihead(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, L, d) → (B, L, H*d)."""
        B, H, L, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * d)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, value_pts: torch.Tensor = None, **kwargs):
        """Forward pass.

        Pads Q, K with zeros to match V's augmented dimension.
        Since zero-padded dims in Q・K^T contribute 0, attention weights
        are mathematically identical and Flash Attention is triggered.

        Args:
            query:      (B, N, d_model)
            key:        (B, M, d_model)
            value:      (B, M, d_model)
            value_pts:  (B, M, 3)

        Returns:
            feat_out:       (B, N, d_model)
            weighted_pts:   (B, N, 3)
        """
        B, N, _ = query.shape
        H = self.n_head
        has_pts = value_pts is not None
        dropout_p = self.dropout_attn.p if self.training else 0.0
        # pad to next multiple of 8 for Flash Attention (≤128, always safe for d_k=64)
        PAD = 8

        # ---- project & reshape → (B, H, *, d) ----
        Q = self._reshape_multihead(self.W_q(query), H)   # (B, H, N, d_k)
        K = self._reshape_multihead(self.W_k(key), H)     # (B, H, M, d_k)
        V = self._reshape_multihead(self.W_v(value), H)   # (B, H, M, d_v)

        if has_pts:
            # Expand points: (B, M, 3) → (B, H, M, 3)
            pts = value_pts.unsqueeze(1).expand(-1, H, -1, -1)
            # V_aug = [V | pts | zeros] → match padded Q/K dims
            V = torch.cat([V, pts], dim=-1)                   # (B, H, M, d_v+3)
            V = F.pad(V, (0, PAD - 3))                        # (B, H, M, d_k+PAD)

        # Zero-pad Q, K to match V dims (does NOT change attention weights)
        Q = F.pad(Q, (0, PAD))                                # (B, H, N, d_k+PAD)
        K = F.pad(K, (0, PAD))                                # (B, H, M, d_k+PAD)
        raw_type = Q.dtype
        if raw_type != torch.bfloat16:
            Q = Q.to(torch.bfloat16)
            K = K.to(torch.bfloat16)
            V = V.to(torch.bfloat16)
        # ---- single SDPA (Flash Attention) ----
        with sdp_kernel(enable_math=False, enable_mem_efficient=False):
            out = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=dropout_p)                  # (B, H, N, d_k+PAD)
            if raw_type != torch.bfloat16:
                out = out.to(raw_type)
        # ---- split features and points ----
        attn_v = out[..., :self.d_v]                           # (B, H, N, d_v)
        if has_pts:
            weighted_pts = out[..., self.d_v:self.d_v + 3].mean(dim=1)  # (B, N, 3)
        else:
            weighted_pts = None

        # ---- output projection (features only) ----
        feat_out = self._merge_multihead(attn_v)               # (B, N, d_model)
        feat_out = self.W_o(feat_out)

        return feat_out, weighted_pts


class PointAugmentedFFN(nn.Module):
    """FFN that separately processes feature and point coordinates.

    Feature path: standard MLP (linear_up → GELU → dropout → linear_down).
    Point path:   configurable projection, mirroring ``project_pts`` in
                  ``ResidualMLPHead``.

    Modes for point_ffn_mode:
        - ``"mlp"``: ``nn.Linear(num_points, num_points, bias=False)``
        - ``"vn"``:  ``VN4Head(num_points)``
        - ``"moe"``: ``MOELayer(emb_dims, num_points, ...)``
    """

    def __init__(self, d_model: int, d_ff: int, num_points: int,
                 dropout: float = 0.1,
                 point_ffn_mode: str = "none",
                 point_ffn_emb_dims: int = 512):
        super().__init__()
        self.d_model = d_model
        self.point_ffn_mode = point_ffn_mode

        # Feature FFN (standard)
        self.feat_w1 = nn.Linear(d_model, d_ff)
        self.feat_w2 = nn.Linear(d_ff, d_model)
        self.feat_dropout = nn.Dropout(dropout)

        # Point FFN (configurable)
        if point_ffn_mode == "mlp":
            self.pts_proj = nn.Sequential(
                nn.Linear(num_points, num_points, bias=False),
            )
        elif point_ffn_mode == "vn":
            from taxpose.nets.vn_dgcnn import VN4Head
            self.pts_proj = VN4Head(num_points)
        elif point_ffn_mode == "moe":
            from taxpose.nets.moe_wab import MOELayer
            self.pts_proj = MOELayer(
                point_ffn_emb_dims, num_points, expert_num=8, top_k=2)
        elif point_ffn_mode == "none":
            self.pts_proj = None
        else:
            raise ValueError(f"Unknown point_ffn_mode: {point_ffn_mode}")

    def forward(self, feat: torch.Tensor, pts: torch.Tensor,
                feat_context: torch.Tensor = None):
        """Forward pass.

        Args:
            feat:          (B, N, C)  features from attention
            pts:           (B, N, 3)  weighted point coordinates
            feat_context:  (B, N, C)  optional embedding for "moe" mode

        Returns:
            feat_out:  (B, N, C)
            pts_out:   (B, N, 3)
        """
        # ---- Feature FFN ----
        feat_out = F.gelu(self.feat_w1(feat))
        feat_out = self.feat_dropout(feat_out)
        feat_out = self.feat_w2(feat_out)

        # ---- Point FFN ----
        if self.pts_proj is not None and pts is not None:
            pts_3n = pts.transpose(1, 2).contiguous()        # (B, 3, N)
            center = pts_3n.mean(dim=2, keepdim=True)
            pts_3n = pts_3n - center
            if self.point_ffn_mode == "moe":
                # MOELayer expects (B, 3, N) and (B, C, N) tuple
                if feat_context is not None:
                    ctx = feat_context.transpose(1, 2).contiguous()  # (B, C, N)
                else:
                    ctx = feat.transpose(1, 2).contiguous()
                pts_out = self.pts_proj((pts_3n, ctx))            # (B, 3, N)
                pts_out = pts_out.transpose(1, 2).contiguous()    # (B, N, 3)
            elif self.point_ffn_mode == "vn":
                # VN4Head expects (B, 3, N)
                # pts_3n = pts.transpose(1, 2).contiguous()
                # center before projection
                pts_out = self.pts_proj(pts_3n)
                pts_out = pts_out + center
                pts_out = pts_out.transpose(1, 2).contiguous()    # (B, N, 3)
            elif self.point_ffn_mode == "mlp":
                # nn.Linear(N, N): expects (B, 3, N)
                # pts_3n = pts.transpose(1, 2).contiguous()
                pts_out = self.pts_proj(pts_3n)
                pts_out = pts_out + center
                pts_out = pts_out.transpose(1, 2).contiguous()
        else:
            pts_out = pts

        return feat_out, pts_out


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("Testing PointAugmentedCrossAttention")
    print("=" * 60)

    B, N, M, C = 2, 64, 128, 256
    n_heads = 4

    # Without return_attn (Flash Attention path)
    attn = PointAugmentedCrossAttention(
        n_head=n_heads, d_model=C, dropout=0.0, return_attn=False).to(device)
    query = torch.randn(B, N, C, device=device)
    key = torch.randn(B, M, C, device=device)
    value = torch.randn(B, M, C, device=device)
    value_pts = torch.randn(B, M, 3, device=device)

    feat_out, weighted_pts = attn(query, key, value, value_pts)
    print(f"feat_out:      {feat_out.shape}")       # (B, N, C)
    print(f"weighted_pts:  {weighted_pts.shape}")   # (B, N, 3)
    assert feat_out.shape == (B, N, C)
    assert weighted_pts.shape == (B, N, 3)
    assert attn.attn is None, "Flash Attention path should not store attn"

    # With return_attn
    attn2 = PointAugmentedCrossAttention(
        n_head=n_heads, d_model=C, dropout=0.0, return_attn=True).to(device)
    feat_out2, weighted_pts2, attn_weights = attn2(query, key, value, value_pts)
    print(f"feat_out2:     {feat_out2.shape}")
    print(f"weighted_pts2: {weighted_pts2.shape}")
    print(f"attn_weights:  {attn_weights.shape}")   # (B, H, N, M)
    assert attn_weights.shape == (B, n_heads, N, M)

    # Without points (pure feature attention)
    feat_out3, weighted_pts3 = attn(query, key, value, None)
    print(f"\nWithout points - feat_out: {feat_out3.shape}, pts: {weighted_pts3}")

    print(f"\n{'=' * 60}")
    print("Testing PointAugmentedFFN")
    print("=" * 60)

    ffn = PointAugmentedFFN(
        d_model=C, d_ff=4*C, num_points=N,
        dropout=0.0, point_ffn_mode="mlp").to(device)
    feat_in = torch.randn(B, N, C, device=device)
    pts_in = torch.randn(B, N, 3, device=device)
    feat_out, pts_out = ffn(feat_in, pts_in)
    print(f"mlp  - feat_out: {feat_out.shape}, pts_out: {pts_out.shape}")
    assert feat_out.shape == (B, N, C)
    assert pts_out.shape == (B, N, 3)

    # "none" mode: no point projection
    ffn_none = PointAugmentedFFN(
        d_model=C, d_ff=4*C, num_points=N,
        dropout=0.0, point_ffn_mode="none").to(device)
    _, pts_none = ffn_none(feat_in, pts_in)
    print(f"none - pts_out (should == pts_in): {torch.allclose(pts_none, pts_in)}")

    print("\n✓ All tests passed!")

