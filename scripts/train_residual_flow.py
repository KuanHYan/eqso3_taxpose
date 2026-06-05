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
from taxpose.nets.transformer_flow import create_network
from taxpose.training.flow_equivariance_training_module_nocentering import (
    EquivarianceTrainingModule,
)
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


@hydra.main(version_base="1.1", config_path="../configs", config_name="train_ndf")
def main(cfg):
    # wandb.init(resume=cfg.resume_ckpt is not None)

    torch.set_float32_matmul_precision("medium")
    TESTING = os.environ.get("PYTEST_CURRENT_TEST", 'False').lower() == "True".lower()

    if cfg.resume_ckpt:
        print("Resuming from checkpoint", cfg.resume_ckpt)
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

    trainer = pl.Trainer(
        logger=False if TESTING else logger,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True) if device_count > 1 else "auto",
        devices=device_count,
        sync_batchnorm=(device_count > 1),
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        # reload_dataloaders_every_n_epochs=1,
        # callbacks=[SaverCallbackModel(), SaverCallbackEmbnnActionAnchor()],
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
                    filename="{epoch}-{step}-{point_loss:.2f}-weights-only",
                    monitor="val_point_loss",
                    mode="min",
                    save_weights_only=True,
                ),
            ]
            if not TESTING else []
        ),
        max_epochs=cfg.training.max_epochs,
        fast_dev_run=20 if TESTING else False,
        precision=cfg.training.precision,
    )
    trainer.print(OmegaConf.to_yaml(cfg, resolve=True))
    trainer.print(f"use {device_count} gpus")

    dm = MultiviewDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.resources.num_workers,
        cfg=cfg.dm,
    )
    dm.setup()

    network = create_network(cfg.model)
    trainer.print(network)
    if cfg.training.lr_scheduler_by_epoch:
        lr_scheduler_total_steps = cfg.training.max_epochs * cfg.training.end_lr_ratio
    else:
        lr_scheduler_total_steps = cfg.training.max_epochs * cfg.training.end_lr_ratio \
            * int(len(dm.train_dataset) / cfg.training.batch_size / device_count)
    cfg.training.lr = cfg.training.lr * device_count
    cfg.training.min_lr = cfg.training.min_lr * device_count    
    lr_scheduler_total_steps = int(lr_scheduler_total_steps)

    # For distributed training, we need to make sure that the learning rate scales with the number of GPUs.
    cfg.training.lr = cfg.training.lr * device_count
    cfg.training.min_lr = cfg.training.min_lr * device_count
    trainer.print(f"real_lr: {cfg.training.lr}, real_min_lr: {cfg.training.min_lr}")
    scheduler_cfg = {
        'scheduler': cfg.training.scheduler,
        'max_steps': lr_scheduler_total_steps,
        'warmup_ratio': cfg.training.warmup_ratio,
        'min_lr': cfg.training.min_lr,
        'by_epoch': cfg.training.lr_scheduler_by_epoch,
    }
    trainer.print(f"lr_scheduler_total_steps: {lr_scheduler_total_steps}, warmup_step: {cfg.training.warmup_ratio*lr_scheduler_total_steps}")
    if cfg.debug:
        from torch.utils.tensorboard.writer import SummaryWriter
        tensorboard_writer = SummaryWriter()
    else:
        tensorboard_writer = None
    model = EquivarianceTrainingModule(
        network,
        lr=cfg.training.lr,
        lr_cfg=scheduler_cfg,
        image_log_period=cfg.training.image_logging_period,
        displace_loss_weight=cfg.training.displace_loss_weight,
        consistency_loss_weight=cfg.training.consistency_loss_weight,
        direct_correspondence_loss_weight=cfg.training.direct_correspondence_loss_weight,
        indirect_correspondence_loss_weight=cfg.training.indirect_correspondence_loss_weight,
        res_smooth_loss_weight=cfg.training.res_smooth_loss_weight,
        start_res_flow_epoch=cfg.training.start_res_flow_epoch,
        weight_normalize=cfg.task.phase.weight_normalize,
        sigmoid_on=cfg.training.sigmoid_on,
        softmax_temperature=cfg.task.phase.softmax_temperature,
        flow_supervision=cfg.training.flow_supervision,
        tr_super_time_ratio=cfg.training.tr_super_start_time_ratio,
        point_cloud_loss=cfg.training.point_cloud_loss,
        tensorboard_writer=tensorboard_writer
    )

    model.cuda()
    model.train()
    if cfg.training.load_from_checkpoint:
        trainer.print("loaded checkpoint from", cfg.training.checkpoint_file)
        model.load_state_dict(
            torch.load(hydra.utils.to_absolute_path(cfg.training.checkpoint_file))[
                "state_dict"
            ]
        )

    else:
        # Might be empty and not have those keys defined.
        # TODO: move this pretraining into the model itself.
        # TODO: figure out if we can get rid of the dictionary and make it null.
        if "pretraining" in cfg.model:
            if cfg.model.pretraining.action.ckpt_path is not None:
                # # Check to see if it's a wandb checkpoint.
                # TODO: need to retrain a few things... checkpoint didn't stick...
                emb_nn_action_state_dict = load_emb_weights(
                    cfg.model.pretraining.action.ckpt_path, cfg.wandb, logger.experiment
                )
                # checkpoint_file_fn = maybe_load_from_wandb(
                #     cfg.pretraining.checkpoint_file_action, cfg.wandb, logger.experiment.run
                # )

                model.model.emb_nn_action.load_state_dict(emb_nn_action_state_dict)
                print(
                    "-----------------------Pretrained EmbNN Action Model Loaded!-----------------------"
                )
                print(
                    "Loaded Pretrained EmbNN Action: {}".format(
                        cfg.model.pretraining.action.ckpt_path
                    )
                )
            if cfg.model.pretraining.anchor.ckpt_path is not None:
                emb_nn_anchor_state_dict = load_emb_weights(
                    cfg.model.pretraining.anchor.ckpt_path, cfg.wandb, logger.experiment
                )
                model.model.emb_nn_anchor.load_state_dict(emb_nn_anchor_state_dict)
                print(
                    "-----------------------Pretrained EmbNN Anchor Model Loaded!-----------------------"
                )
                print(
                    "Loaded Pretrained EmbNN Anchor: {}".format(
                        cfg.model.pretraining.anchor.ckpt_path
                    )
                )
    if not cfg.eval:
        model.eval()
        trainer.validate(model, dm, ckpt_path=resume_ckpt)
        model.train()
        trainer.fit(model, dm, ckpt_path=resume_ckpt)
    model.eval()
    trainer.validate(model, dm, ckpt_path=resume_ckpt)

    # Print he run id of the current run
    print("Run ID: {} ".format(logger.experiment.id))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    # torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
