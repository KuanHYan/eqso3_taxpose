from embedding import create_embedding_network
from head import create_head, HeadConfig
from transformer_flow import (
    ModelConfig,
    ResidualFlow_DiffEmbTransformer,
    ResidualFlowDiffEmbTransformerConfig,
    TwoStageFlowTransformer
)
from cascade_transformer import (
    CascadeFlowTransformer,
    RecurrentFlowTransformer,
)
from typing import cast


def create_network(cfg: ModelConfig):
    # Create the network
    if cfg.model_type == "residual_flow_diff_emb_transformer":
        r_cfg = cast(ResidualFlowDiffEmbTransformerConfig, cfg)
        network = ResidualFlow_DiffEmbTransformer(
            encoder_cfg=r_cfg.encoder,
            head_cfg=r_cfg.head,
            cycle=r_cfg.cycle,
            center_feature=r_cfg.center_feature,
            freeze_embnn=r_cfg.freeze_embnn,
            return_attn=r_cfg.return_attn,
            multilaterate=r_cfg.multilaterate,
            mlat_sample=r_cfg.mlat_sample,
            mlat_nkps=r_cfg.mlat_nkps,
            feature_channels=r_cfg.feature_channels,
            conditional=r_cfg.conditional,
            dropout=r_cfg.dropout,
            pos_encoding=r_cfg.pos_encoding,
            n_blocks=r_cfg.n_blocks,
            attn_mode=r_cfg.attn_mode,
            fine_tune=getattr(cfg, 'fine_tune', False),
            weight_beta=getattr(cfg, 'weight_beta', 0.5),
            knn_mode=getattr(cfg, 'knn_mode', 'emb'),
            point_augmented=getattr(cfg, 'point_augmented', False),
            point_ffn_mode=getattr(cfg, 'point_ffn_mode', 'none'),
        )
    elif cfg.model_type == "two_stage_flow_transformer":
        r_cfg = cast(ResidualFlowDiffEmbTransformerConfig, cfg)
        network = TwoStageFlowTransformer(
            encoder_cfg=r_cfg.encoder,
            head_cfg=r_cfg.head,
            knn_mode=getattr(cfg, 'knn_mode', 'pt'),
            cycle=r_cfg.cycle,
            center_feature=r_cfg.center_feature,
            freeze_embnn=r_cfg.freeze_embnn,
            return_attn=r_cfg.return_attn,
            multilaterate=r_cfg.multilaterate,
            mlat_sample=r_cfg.mlat_sample,
            mlat_nkps=r_cfg.mlat_nkps,
            feature_channels=r_cfg.feature_channels,
            conditional=r_cfg.conditional,
            dropout=r_cfg.dropout,
            pos_encoding=r_cfg.pos_encoding,
            n_blocks=r_cfg.n_blocks,
            attn_mode=r_cfg.attn_mode,
            num_refine_steps=r_cfg.num_refine_steps,
            refine_hidden_dim=r_cfg.refine_hidden_dim,
            fine_tune_trans=r_cfg.fine_tune,
            point_augmented=getattr(cfg, 'point_augmented', False),
            point_ffn_mode=getattr(cfg, 'point_ffn_mode', 'none'),
        )
    elif cfg.model_type == "cascade_flow_transformer":
        r_cfg = cast(ResidualFlowDiffEmbTransformerConfig, cfg)
        network = CascadeFlowTransformer(
            encoder_cfg=r_cfg.encoder,
            stage_num=r_cfg.stage_num,
            num_refine_steps=r_cfg.num_refine_steps,
            head_cfg=r_cfg.head,
            cycle=r_cfg.cycle,
            center_feature=r_cfg.center_feature,
            freeze_embnn=r_cfg.freeze_embnn,
            return_attn=r_cfg.return_attn,
            multilaterate=r_cfg.multilaterate,
            mlat_sample=r_cfg.mlat_sample,
            mlat_nkps=r_cfg.mlat_nkps,
            feature_channels=r_cfg.feature_channels,
            conditional=r_cfg.conditional,
            dropout=r_cfg.dropout,
            pos_encoding=r_cfg.pos_encoding,
            n_blocks=r_cfg.n_blocks,
            attn_mode=r_cfg.attn_mode,
            refine_hidden_dim=r_cfg.refine_hidden_dim,
            fine_tune_trans=r_cfg.fine_tune,
            knn_mode=getattr(cfg, 'knn_mode', 'pt'),
            point_augmented=getattr(cfg, 'point_augmented', False),
            point_ffn_mode=getattr(cfg, 'point_ffn_mode', 'none'),
            stage_emb_cat=getattr(cfg, 'stage_emb_cat', 'mlp'),
        )
    elif cfg.model_type == "recurrent_flow_transformer":
        r_cfg = cast(ResidualFlowDiffEmbTransformerConfig, cfg)
        network = RecurrentFlowTransformer(
            encoder_cfg=r_cfg.encoder,
            head_cfg=r_cfg.head,
            num_iterations=r_cfg.stage_num,
            stage_emb_cat=getattr(cfg, 'stage_emb_cat', 'mlp'),
            cycle=r_cfg.cycle,
            center_feature=r_cfg.center_feature,
            freeze_embnn=r_cfg.freeze_embnn,
            return_attn=r_cfg.return_attn,
            multilaterate=r_cfg.multilaterate,
            mlat_sample=r_cfg.mlat_sample,
            mlat_nkps=r_cfg.mlat_nkps,
            feature_channels=r_cfg.feature_channels,
            conditional=r_cfg.conditional,
            dropout=r_cfg.dropout,
            pos_encoding=r_cfg.pos_encoding,
            n_blocks=r_cfg.n_blocks,
            attn_mode=r_cfg.attn_mode,
            knn_mode=getattr(cfg, 'knn_mode', 'emb'),
            point_augmented=getattr(cfg, 'point_augmented', False),
            point_ffn_mode=getattr(cfg, 'point_ffn_mode', 'none'),
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")

    return network
