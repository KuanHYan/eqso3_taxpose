import json
import os

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from taxpose.datasets.pretraining_point_cloud_data_module import (
    PretrainingMultiviewDataModule,
)
from taxpose.nets.transformer_flow import EquivariantFeatureEmbeddingNetwork
from taxpose.utils.load_model import get_weights_path
from taxpose.training.equivariant_feature_pretraining_vae import (
    EquivariancePreTrainingModule,
)
from taxpose.utils.dup_stdout_manager import DupStdoutFileManager
from taxpose.nets.raw_dgcnn import DGCNN_VAE

@hydra.main(version_base="1.1", config_path="../configs", config_name="pretraining")
def main(cfg):
    file_suffix = cfg.job_type
    full_log_name = f"train_{file_suffix}"
    # if os.path.exists(os.path.join(cfg.output_dir, f"{full_log_name}.log")):
    #     raise ValueError(f"{full_log_name} already exists!")
    with DupStdoutFileManager(
            os.path.join(cfg.output_dir, f"{full_log_name}.log")
    ) as _:
        print(
            json.dumps(
                omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
                sort_keys=True,
                indent=4,
            )
        )

    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")

    pl.seed_everything(cfg.seed)

    TESTING = os.environ.get("PYTEST_CURRENT_TEST", 'False') == "True"
    
    if cfg.resume_ckpt:
        print("Resuming from checkpoint")
        print(cfg.resume_ckpt)
        resume_ckpt = get_weights_path(cfg.resume_ckpt, cfg.wandb)
        resume_ckpt = os.path.join(cfg.log_dir, resume_ckpt)
        # Resume the wandb run
        if cfg.resume_ckpt.startswith(cfg.wandb.entity):
            # Get the run_id from the checkpoint
            resume_run_id = cfg.resume_ckpt.split("/")[2].split("-")[1].split(":")[0]
        elif cfg.wandb.run_id_override is not None:
            resume_run_id = cfg.wandb.run_id_override
        else:
            resume_run_id = None

    else:
        resume_ckpt = None
        resume_run_id = None

    logger = WandbLogger(
        name=cfg.wandb.name,
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        save_dir=cfg.wandb.save_dir,
        job_type=cfg.job_type,
        offline=cfg.wandb.offline,
        save_code=True,
        log_model=not cfg.wandb.offline,
        id=cfg.wandb.name if resume_run_id is None else resume_run_id,
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
    )

    trainer = pl.Trainer(
        logger=False if TESTING else logger,
        accelerator="gpu",
        devices=[0],
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        max_epochs=cfg.training.epochs,
        precision=cfg.training.precision,
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
                    monitor="train_loss",
                    mode="min",
                    save_weights_only=True,
                ),
            ]
            if not TESTING else []
        ),
        fast_dev_run=5 if TESTING else False,
    )

    dm = PretrainingMultiviewDataModule(
        cfg=cfg.dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.resources.num_workers,
    )

    network = DGCNN_VAE(cfg.encoder, cfg.encoder.pos_encoding)
    if TESTING:
        print(network)
    scheduler_cfg = {
        'scheduler': cfg.training.scheduler,
        'max_steps': int(cfg.training.epochs * cfg.dataset.train_dset.data_size),
        'warmup_ratio': 0.01,
        'by_epoch': False
    }
    model = EquivariancePreTrainingModule(
        network,
        lr=cfg.training.lr,
        image_log_period=cfg.training.image_logging_period,
        l2_reg_weight=cfg.training.l2_reg_weight,
        normalize_features=cfg.training.normalize_features,
        temperature=cfg.training.temperature,
        con_weighting=cfg.training.con_weighting,
        lr_cfg=scheduler_cfg,
    )

    trainer.fit(model, dm, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")

    main()
