import os

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from taxpose.datasets.point_cloud_data_module import MultiviewDataModule
from taxpose.training.rl_fine_tune import RLTrainingModule
from taxpose.nets.RL_policy import PolicyModel
from taxpose.utils.load_model import get_weights_path

def load_emb_weights(checkpoint_reference, wandb_cfg=None, run=None):
    if checkpoint_reference.startswith(wandb_cfg.entity):
        artifact_dir = os.path.join(wandb_cfg.artifact_dir, checkpoint_reference)
        if run is None or not isinstance(run, wandb.sdk.wandb_run.Run):
            # Download without a run
            api = wandb.Api()
            artifact = api.artifact(checkpoint_reference, type="model")
        else:
            artifact = run.use_artifact(checkpoint_reference)
        checkpoint_path = artifact.get_path("model.ckpt").download(root=artifact_dir)
        weights = torch.load(checkpoint_path)["state_dict"]
        # remove "model.emb_nn" prefix from keys
        weights = {k.replace("model.emb_nn.", ""): v for k, v in weights.items()}
        return weights
    else:
        net_ws = torch.load(hydra.utils.to_absolute_path(checkpoint_reference))[
            "state_dict"  # BUG: saved checkpoints don't contain key "embnn_state_dict"
        ]
        # BUG: the dict keys of saved checkpoints using original pretrain_embedding.py is:
        # ['model.emb_nn.conv1.weight', ..., 'model.emb_nn.bn1.weight', ...].
        # However, needed dict keys in tax-pose model are:
        # ['conv1.weight', ..., 'bn1.weight', ...].
        if "model.emb_nn" in list(net_ws.keys())[0]:
            weights = {k.replace("model.emb_nn.", ""): v for k, v in net_ws.items()}
        elif "model.encoder" in list(net_ws.keys())[0]:
            weights = {k.replace("model.", ""): v for k, v in net_ws.items()}
        else:
            raise ValueError("Unknown checkpoint format")
        return weights


def maybe_load_from_wandb(checkpoint_reference, wandb_cfg, run):
    if checkpoint_reference.startswith(wandb_cfg.entity):
        # download checkpoint locally (if not already cached)
        artifact_dir = wandb_cfg.artifact_dir
        artifact = run.use_artifact(checkpoint_reference)
        ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    else:
        ckpt_file = checkpoint_reference
    return ckpt_file


def set_cfg_fpr_debug(cfg):
    torch.cuda.set_device(0)
    OmegaConf.update(cfg, "job_type", "rl_tune")
    OmegaConf.update(cfg, "data_root", "/home/yan/pose_estimation/taxpose/data/ideal_pair_models")
    OmegaConf.update(cfg, "training.max_epochs", 500)
    OmegaConf.update(cfg, "training.check_val_every_n_epoch", 1)
    OmegaConf.update(cfg, "training.batch_size", 8)
    OmegaConf.update(cfg, "training.lr", 1e-6)
    OmegaConf.update(cfg, "training.min_lr", 1e-7)
    OmegaConf.update(cfg, "training.warmup_ratio", 0.1)
    OmegaConf.update(cfg, "training.precision", '32')
    OmegaConf.update(cfg, "training.num_gpus", 1)
    OmegaConf.update(cfg, "training.accumulate_grad_batches", 2)
    OmegaConf.update(cfg, "training.scheduler", "linear")
    OmegaConf.update(cfg, "dm.train_dset.demo_dset.num_demo", 600)
    OmegaConf.update(cfg, "dm.train_dset.dataset_size", 640)
    OmegaConf.update(cfg, "dm.train_dset.anchor_rot_sample_method", "axis_angle")
    OmegaConf.update(cfg, "dm.train_dset.anchor_rotation_variance", 3.141592653589793)
    OmegaConf.update(cfg, "model.freeze_embnn", True)
    OmegaConf.update(cfg, "model.dropout", 0.1)
    OmegaConf.update(cfg, "model.n_blocks", 1)
    OmegaConf.update(cfg, "model.cycle", True)
    OmegaConf.update(cfg, "model.encoder.name", "raw_dgcnn")
    OmegaConf.update(cfg, "model.encoder.emb_dims", 512)
    OmegaConf.update(cfg, "model.encoder.norm", "BN")
    OmegaConf.update(cfg, "model.encoder.output_num", 1024)
    OmegaConf.update(cfg, "model.encoder.dropout", 0.1)
    OmegaConf.update(cfg, "model.head.head_type", "rl_residual")
    OmegaConf.update(cfg, "model.head.project_corrs", True)
    OmegaConf.update(cfg, "model.head.project_corrs_mode", "moe")
    OmegaConf.update(cfg, "model.head.norm", "LN")
    OmegaConf.update(cfg, "model.head.head_bias", False)
    OmegaConf.update(cfg, "model.head.residual_on", True)
    OmegaConf.update(cfg, "model.head.pred_weight", True)
    OmegaConf.update(cfg, "model.head.reparam", False)
    OmegaConf.update(cfg, "rl.group", 8)
    OmegaConf.update(cfg, "rl.update_base_every", 100)
    OmegaConf.update(cfg, "rl.kl_coef", 0.02)
    OmegaConf.update(cfg, "rl.clip_eps", 0.2)
    OmegaConf.update(cfg, "wandb.name", "debug")
    OmegaConf.update(cfg, "wandb.offline", True)
    OmegaConf.update(cfg, "debug", False)
    OmegaConf.update(cfg, "eval", False)
    OmegaConf.update(cfg, "rl.reward_model_path", "/home/yan/pose_estimation/taxpose/trained_models/reward_w.ckpt")
    OmegaConf.update(cfg, "rl.base_model_path", "/home/yan/pose_estimation/taxpose/logs/train_taxpose/2026-06-10/13-05-52/checkpoints/last.ckpt")
    return cfg

@hydra.main(version_base="1.1", config_path="../configs", config_name="train_ndf")
def main(cfg):
    if __name__ == "__main__":
        print(OmegaConf.to_yaml(cfg, resolve=True))

    torch.set_float32_matmul_precision("medium")
    TESTING = os.environ.get("PYTEST_CURRENT_TEST", 'False').lower() == "True".lower()

    ## debug
    # cfg = set_cfg_fpr_debug(cfg)

    if cfg.resume_ckpt:
        print("Resuming from checkpoint")
        print(cfg.resume_ckpt)
        resume_ckpt = get_weights_path(cfg.resume_ckpt, cfg.wandb)
        # 判断resume_ckpt是否绝对路径
        if not resume_ckpt.startswith('/'):
            resume_ckpt = hydra.utils.to_absolute_path(resume_ckpt)

        # Resume the wandb run
        if cfg.resume_ckpt.startswith(cfg.wandb.entity):
            # Get the run_id from the checkpoint
            resume_run_id = cfg.resume_ckpt.split("/")[2].split("-")[1].split(":")[0]
        elif cfg.wandb.run_id_override is not None:
            resume_run_id = cfg.wandb.run_id_override
        elif cfg.wandb.name is not None:
            resume_run_id = cfg.wandb.name
        else:
            resume_run_id = None

    else:
        resume_ckpt = None
        resume_run_id = None

    print("Resume run id:", resume_run_id)
    device_count = min(torch.cuda.device_count(), cfg.training.num_gpus)
    if device_count == 1:
        pl.seed_everything(cfg.seed)
    logger = WandbLogger(
        name=cfg.wandb.name,
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        save_dir=cfg.wandb.save_dir,
        job_type=cfg.job_type,
        save_code=not (cfg.wandb.offline or TESTING or cfg.debug),
        log_model=not (cfg.wandb.offline or TESTING or cfg.debug),
        id=cfg.wandb.name if resume_run_id is None else resume_run_id,
        offline=cfg.wandb.offline or TESTING or cfg.debug,
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
    )
    # logger.log_hyperparams(cfg)
    # logger.log_hyperparams({"working_dir": os.getcwd()})
    trainer = pl.Trainer(
        logger=False if TESTING else logger,
        accelerator="auto",
        strategy=DDPStrategy(find_unused_parameters=True) if device_count > 1 else "auto",
        devices=device_count,
        sync_batchnorm=False,
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        callbacks=(
            [
                # This checkpoint callback saves the latest model during training, i.e. so we can resume if it crashes.
                # It saves everything, and you can load by referencing last.ckpt.
                ModelCheckpoint(
                    dirpath=cfg.lightning.checkpoint_dir,
                    filename="{epoch}-{step}",
                    monitor="step",
                    mode="max",
                    save_weights_only=False,
                    save_last=True,
                    every_n_epochs=1,
                ),
                # This checkpoint will get saved to WandB. The Callback mechanism in lightning is poorly designed, so we have to put it last.
                ModelCheckpoint(
                    dirpath=cfg.lightning.checkpoint_dir,
                    filename="{epoch}-{step}-{train_loss:.2f}-weights-only",
                    monitor="val_point_loss",
                    mode="min",
                    save_weights_only=True,
                ),
            ]
            if not TESTING else []
        ),
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        max_epochs=cfg.training.max_epochs,
        fast_dev_run=20 if TESTING else False,
        precision=cfg.training.precision,
    )
    validationer = pl.Trainer(
        logger=False if TESTING else logger,
        accelerator="auto",
        devices=1,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )
    trainer.print(f"use {device_count} GPUs")
    dm = MultiviewDataModule(
        trainbatch_size=cfg.dm.train_mini_batch_size,
        valbatch_size=cfg.dm.val_mini_batch_size * 2,
        num_workers=cfg.resources.num_workers,
        cfg=cfg.dm,
    )

    dm.setup()
    cfg.rl.reward_model_path = hydra.utils.to_absolute_path(cfg.rl.reward_model_path)
    network = PolicyModel(
        cfg.model.encoder,
        cfg.model.head,
        cfg.rl.reward_model_path,
        cfg.model.cycle,
        center_feature=cfg.model.center_feature,
        freeze_embnn=cfg.model.freeze_embnn,
        return_attn=cfg.model.return_attn,
        dropout=cfg.model.dropout,
        pos_encoding=cfg.model.pos_encoding,
        group=cfg.rl.group,
        n_blocks=int(cfg.model.n_blocks),
        attn_mode=cfg.model.attn_mode,
        manual_reawrd=True
    )

    trainer.print(network)
    device_count = device_count * cfg.training.accumulate_grad_batches
    if cfg.training.lr_scheduler_by_epoch:
        lr_scheduler_total_steps = cfg.training.max_epochs * cfg.training.end_lr_ratio
    else:
        lr_scheduler_total_steps = cfg.training.max_epochs * cfg.training.end_lr_ratio \
            * int(len(dm.train_dataset) / cfg.training.batch_size / device_count)
    lr_scheduler_total_steps = int(lr_scheduler_total_steps)
    cfg.training.lr = cfg.training.lr * device_count
    cfg.training.min_lr = cfg.training.min_lr * device_count
    trainer.print(f"real_lr: {cfg.training.lr}, real_min_lr: {cfg.training.min_lr}")
    scheduler_cfg = {
        'scheduler': cfg.training.scheduler,
        'max_steps': lr_scheduler_total_steps,
        'warmup_ratio': cfg.training.warmup_ratio,
        'min_lr': cfg.training.min_lr,
        'by_epoch': cfg.training.lr_scheduler_by_epoch,
        "weight_decay": cfg.training.weight_decay,
    }
    trainer.print(f"lr_scheduler_total_steps: {lr_scheduler_total_steps}, warmup_step: {cfg.training.warmup_ratio*lr_scheduler_total_steps}")
    if cfg.debug:
        from torch.utils.tensorboard.writer import SummaryWriter
        tensorboard_writer = SummaryWriter()
    else:
        tensorboard_writer = None
    model = RLTrainingModule(
        network,
        lr=cfg.training.lr,
        lr_cfg=scheduler_cfg,
        image_log_period=cfg.training.image_logging_period,
        flow_supervision=cfg.training.flow_supervision,
        kl_coef=cfg.rl.kl_coef,
        clip_eps=cfg.rl.clip_eps,
        update_base_every=cfg.rl.update_base_every,
        grpo_iter=cfg.rl.grpo_iter,
        optimization_mode=cfg.training.optimization_mode,
        tensorboard_writer=tensorboard_writer
    )

    model.cuda()
    model.train()
    base_model_path = hydra.utils.to_absolute_path(cfg.rl.base_model_path)
    trainer.print(f"loaded base model from: {base_model_path}")
    model.load_state_dict(
        torch.load(base_model_path)[
            "state_dict"
        ],
    )
    if not cfg.eval:
        model.train()
        trainer.fit(model, dm, ckpt_path=resume_ckpt)

    model.eval()
    validationer.validate(model, dm, ckpt_path=resume_ckpt)

    # Print he run id of the current run
    print("Run ID: {} ".format(logger.experiment.id))


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
