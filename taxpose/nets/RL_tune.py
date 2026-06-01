import torch
import torch.nn as nn
import torch.nn.functional as F

from taxpose.nets.transformer_flow import create_embedding_network
from taxpose.nets.transformer_flow_pm import CustomTransformer


class RewardModel(nn.Module):
    def __init__(
        self,
        encoder_cfg,
        cycle=True,
        center_feature=False,
        feature_channels=0,  # Number of extra channels we'll pass into the network.
        dropout=0.1,
        pos_encoding=False,
    ):
        super(RewardModel, self).__init__()
        self.cycle = cycle
        self.feature_channels = feature_channels

        self.emb_nn_action = create_embedding_network(encoder_cfg)
        self.emb_nn_anchor = create_embedding_network(encoder_cfg)
        emb_dims = encoder_cfg.emb_dims

        self.center_feature = center_feature
        self.transformer_action = nn.Transformer(
            d_model=emb_dims,
            nhead=emb_dims//64,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=emb_dims*4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dtype=torch.float32
        )
        self.transformer_anchor = nn.Transformer(
            d_model=emb_dims,
            nhead=emb_dims//64,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=emb_dims*4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dtype=torch.float32
        )

        self.reward_head_gate = nn.Sequential(
            nn.LayerNorm(emb_dims),
            nn.Linear(emb_dims, 4 * emb_dims),
            nn.GELU(),
            nn.Linear(4 * emb_dims, 1),
            nn.Sigmoid()
        )

        self.reward_head = nn.Linear(emb_dims, 1)

        self.pos_encoding = pos_encoding
        # if pos_encoding:
        #     self.pos_encoder = ManualPointWiseGemoFea(True, emb_dims)

    def forward(self, *input, return_total_reward=False):
        action_points = input[0].permute(0, 2, 1)[:, :3]  # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)[:, :3]

        # action_points_dmean = action_points - action_points.mean(dim=2, keepdim=True)
        # anchor_points_dmean = anchor_points - anchor_points.mean(dim=2, keepdim=True)

        act_down_sample, anch_down_sample = None, None
        # with torch.set_grad_enabled(not self.freeze_embnn):
        action_embedding = self.emb_nn_action(action_points)
        if isinstance(action_embedding, tuple):
            action_embedding, pts = action_embedding
            act_down_sample = pts + action_points.mean(dim=2, keepdim=True)
            act_down_sample = act_down_sample

        anchor_embedding = self.emb_nn_anchor(anchor_points)
        if isinstance(anchor_embedding, tuple):
            anchor_embedding, pts = anchor_embedding
            anch_down_sample = pts + anchor_points.mean(dim=2, keepdim=True)
            anch_down_sample = anch_down_sample

        action_embedding = F.normalize(action_embedding, dim=1)
        anchor_embedding = F.normalize(anchor_embedding, dim=1)

        # tilde_phi, phi are both B,512,N
        # Get the new cross-attention embeddings.
        if self.pos_encoding:
            action_pt_pos = self.pos_encoder(action_points)  # B,C,N
            anchor_pt_pos = self.pos_encoder(anchor_points)
            action_embedding += F.normalize(action_pt_pos)
            anchor_embedding += F.normalize(anchor_pt_pos)

        action_embedding = action_embedding.permute(0, 2, 1)
        anchor_embedding = anchor_embedding.permute(0, 2, 1)

        action_embedding_tf = self.transformer_action(
            anchor_embedding, action_embedding
        )

        action_reward_gate = self.reward_head_gate(
            action_embedding_tf
        )  # [B, N, 1]
        action_reward = self.reward_head(
            (action_embedding_tf * action_reward_gate).sum(dim=1)  # [B, D]
        )  # [B, 1]

        outputs = {"action_reward": action_reward}

        if self.cycle:
            anchor_embedding_tf = self.transformer_anchor(
                action_embedding, anchor_embedding
            )
            anch_reward_gate = self.reward_head_gate(
                anchor_embedding_tf
            )  # [B, N, 1]
            ancho_reward = self.reward_head(
                (anchor_embedding_tf * anch_reward_gate).sum(dim=1)  # [B, D]
            )  # [B, 1]

            outputs.update(
                {"anchor_reward": ancho_reward}
            )
            if return_total_reward:
                return ancho_reward * 0.5 + action_reward * 0.5
        return outputs


if __name__ == "__main__":
    from taxpose.nets.raw_dgcnn import DGCNNArgs

    p1 = torch.randn(2, 3, 1024)
    p2 = torch.randn(2, 3, 1024)
    encoder_args = DGCNNArgs(
        name="raw_dgcnn",
        emb_dims=512,
        knn=2,
    )
    model = RewardModel(
        encoder_args,
        cycle=True,
        dropout=0.1
    )
    y = model(p1, p2)
    print(y.keys(), y["anchor_reward"].shape)