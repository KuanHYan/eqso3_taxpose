import pytorch_lightning as pl
import torch
import wandb
from torchvision.transforms import ToTensor
from torch.optim import lr_scheduler
from taxpose.utils.lr import MilestoneScheduler, LinearAnnealingWarmup
to_tensor = ToTensor()


class PointCloudTrainingModule(pl.LightningModule):
    def __init__(
            self, model=None, lr=1e-3, min_lr=1e-4,
            scheduler: str = 'constant', max_steps: int = 100,
            warmup_ratio: float = 0.1, by_epoch: bool = True,
            tensorboard_writer=None, image_log_period=500,
            ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.min_lr = min_lr
        self.image_log_period = image_log_period
        self.global_val_step = 0
        self.lr_scheduler = scheduler
        self.by_epoch = by_epoch
        self.end_lr_steps = max_steps
        self.warmup_steps = int(warmup_ratio * self.end_lr_steps)
        if tensorboard_writer is not None:
            self.tensorboard_writer = tensorboard_writer
            self.shared_params = list(self.model.parameters())

    def module_step(self, batch, batch_idx):
        raise NotImplementedError("module_step must be implemented by child class")
        return loss, log_values

    def visualize_results(self, batch, batch_idx):
        return {}

    def on_train_end(self) -> None:
        if getattr(self, "tensorboard_writer", None) is not None:
            self.tensorboard_writer.flush()
        return super().on_train_end()

    # def on_after_backward(self) -> None:
    #     # 尝试获取self.writer，如果没有或者是None，则返回
    #     if getattr(self, "tensorboard_writer", None) is None:
    #         return super().on_after_backward()
    #     for name, param in self.model.named_parameters():
    #         if param.grad is not None:
    #             self.tensorboard_writer.add_histogram(
    #                 f"grad/{name}", param.grad, self.global_step)
    #             grad_norm = param.grad.norm(2)
    #             weight_norm = param.data.norm(2)
    #             self.tensorboard_writer.add_scalar(
    #                 f"grad_norm/{name}", grad_norm, self.global_step)
    #             self.tensorboard_writer.add_scalar(
    #                 f"Ratio (Grad/Weight)/{name}", grad_norm / (weight_norm + 1e-8),
    #                 self.global_step
    #             )
    #     return super().on_after_backward()

    def on_before_zero_grad(self, optimizer) -> None:
        if getattr(self, "tensorboard_writer", None) is None:
            return
        grad_list = []
        for l in self._ori_loss:
            optimizer.zero_grad()
            l.backward(retain_graph=True)
            grad_A = [p.grad.clone().view(-1) for p in self.shared_params if p.grad is not None]
            grad_A = torch.cat(grad_A)   # 展平为一维向量
            grad_list.append(grad_A)
        # 3. 计算余弦相似度
        point_loss_grad, smoothness_loss_grad, dense_loss_grad = grad_list
        cos_sim_pc_vpc = torch.dot(point_loss_grad, dense_loss_grad) / (torch.norm(point_loss_grad) * torch.norm(dense_loss_grad) + 1e-8)
        cos_sim_pc_sm = torch.dot(point_loss_grad, smoothness_loss_grad) / (torch.norm(point_loss_grad) * torch.norm(smoothness_loss_grad) + 1e-8)
        cos_sim_vpc_sm = torch.dot(dense_loss_grad, smoothness_loss_grad) / (torch.norm(dense_loss_grad) * torch.norm(smoothness_loss_grad) + 1e-8)
        self.tensorboard_writer.add_scalar("cos_sim_pc_vpc", cos_sim_pc_vpc, self.global_step)
        self.tensorboard_writer.add_scalar("cos_sim_pc_sm", cos_sim_pc_sm, self.global_step)
        self.tensorboard_writer.add_scalar("cos_sim_pc_vpc_sm", cos_sim_vpc_sm, self.global_step)
        return


    def training_step(self, batch, batch_idx):
        loss, log_values = self.module_step(batch, batch_idx)
        if isinstance(loss, tuple):
            self._ori_loss = loss
            loss = sum(loss)
        for key, val in log_values.items():
            self.log(key, val, logger=True)

        if (self.global_step % self.image_log_period) == 0:
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if isinstance(val, wandb.Object3D) and wandb.run is not None:
                    wandb.log(
                        {
                            key: val,
                            "trainer/global_step": self.global_step,
                        }
                    )
                elif self.logger is not None:
                    self.logger.log_image(
                        key,
                        images=[val],  # self.global_step
                    )
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_values = self.module_step(batch, batch_idx)
        if isinstance(loss, tuple):
            self._ori_loss = loss
            loss = sum(loss)
        for key, val in log_values.items():
            self.log("val_" + key, val, logger=True)

        if (self.global_val_step % self.image_log_period) == 0:
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if isinstance(val, wandb.Object3D) and wandb.run is not None:
                    wandb.log(
                        {
                            "val_" + key: val,
                            "trainer/global_step": self.global_val_step,
                        }
                    )
                elif self.logger is not None:
                    self.logger.log_image(
                        "val_" + key,
                        images=[val],  # self.global_val_step
                    )
        self.global_val_step += 1

        self.log("val_loss", loss, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, log_values = self.module_step(batch, batch_idx)
        if isinstance(loss, tuple):
            self._ori_loss = loss
            loss = sum(loss)
        for key, val in log_values.items():
            self.log(key, val, logger=True)

        if (self.global_step % self.image_log_period) == 0:
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if isinstance(val, wandb.Object3D) and wandb.run is not None:
                    wandb.log(
                        {
                            "test_" + key: val,
                        }
                    )
                elif self.logger is not None:
                    self.logger.log_image("test_" + key, val)

        self.log("test_loss", loss, logger=True)
        return loss

    def configure_optimizers(self):
        # return example:
        # {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": ReduceLROnPlateau(optimizer, ...),
        #         "monitor": "metric_to_track",
        #         "frequency": "indicates how often the metric is updated",
        #         # If "monitor" references validation metrics, then "frequency" should be set to a
        #         # multiple of "trainer.check_val_every_n_epoch".
        #     },
        # }
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self._optimizer_ = optimizer
        if self.warmup_steps <= 0:
            return optimizer
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        if self.lr_scheduler == 'constant':
            milestones = [self.end_lr_steps]
            scheduler = MilestoneScheduler(
                optimizer,
                milestones=milestones,
                gamma=1.0,
                max_lr=self.lr,
                min_lr=self.min_lr,
                warmup_steps=self.warmup_steps,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch' if self.by_epoch else 'step'}
            }
        elif self.lr_scheduler == 'milestone':
            milestones = [int(self.end_lr_steps * stone) for stone in [0.5, 0.75, 0.9]]
            scheduler = MilestoneScheduler(
                optimizer,
                milestones=milestones,
                gamma=0.5,
                max_lr=self.lr,
                min_lr=self.min_lr,
                warmup_steps=self.warmup_steps,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch' if self.by_epoch else 'step'}
            }
        elif self.lr_scheduler == 'linear':
            scheduler = LinearAnnealingWarmup(
                optimizer,
                total_steps=self.end_lr_steps,
                max_lr=self.lr,
                min_lr=self.min_lr,
                warmup_steps=self.warmup_steps,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch' if self.by_epoch else 'step'}
            }
        else:
            raise ValueError(f'Invalid lr_scheduler: {self.lr_scheduler}, please choose from constant, milestone, linear')
