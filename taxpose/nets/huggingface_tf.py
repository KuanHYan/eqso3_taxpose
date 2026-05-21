import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


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
    def __init__(self, n_head, d_model, dropout=0.1, project_bias=False):
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

    def forward(self, query, key, value, mask=None):
        x, self.attn = self.attn_net.forward(
            query, key, value,
            need_weights=True, average_attn_weights=False
        )

        return x


if __name__ == "__main__":
    model = Transformer(emb_dims=256, n_blocks=1, dropout=0.1, ff_dims=1024, n_heads=4, return_attn=True, bidirectional=False)
    model.eval()
    act_emb = torch.randn(2, 256, 64)
    tgt_emb = torch.randn(2, 256, 128)
    outputs = model.forward(act_emb, tgt_emb)
    print(outputs["src_embedding"].shape)
    print(outputs["src_attn"].shape)
    print(model.get_attn_scores(act_emb, tgt_emb, seq_dim=2).shape)
