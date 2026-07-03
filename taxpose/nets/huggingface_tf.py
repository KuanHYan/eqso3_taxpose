import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


def get_graph_feature(feas, idx):
    """
    Args:
        feas: B, N, C
        idx: B, N, K
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
                 project_bias=False, need_weights=True):
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
                 project_bias=False, need_weights=True):
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


if __name__ == "__main__":
    # model = Transformer(emb_dims=256, n_blocks=1, dropout=0.1, ff_dims=1024, n_heads=4, return_attn=True, bidirectional=False)
    model = LinearMHAttention(n_head=4, d_model=256, dropout=0.1, project_bias=False)
    model.eval()
    act_emb = torch.randn(2, 64, 256)
    tgt_emb = torch.randn(2, 128, 256)
    outputs = model.forward(act_emb, tgt_emb, tgt_emb, None)
    print(outputs.shape)
    print(model.attn.shape)
    print(model.attn[0])
