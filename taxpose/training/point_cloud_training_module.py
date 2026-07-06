import pytorch_lightning as pl
import torch
import wandb
from torchvision.transforms import ToTensor
from taxpose.utils.lr import MilestoneScheduler, LinearAnnealingWarmup
from taxpose.nets.head import TransformerHead
to_tensor = ToTensor()


class PointCloudTrainingModule(pl.LightningModule):
    def __init__(
            self, model=None, lr=1e-3, min_lr=1e-4,
            scheduler: str = 'constant', max_steps: int = 100,
            warmup_ratio: float = 0.1, weight_decay=1e-4,
            by_epoch: bool = True,
            debug=False, image_log_period=500,
            optimization_mode: str = 'auto',
            # 分层学习率系数 (emb/backbone/head/gru 相对 base_lr 的比例)
            emb_lr_ratio: float = 0.1,
            backbone_lr_ratio: float = 1.0,
            head_lr_ratio: float = 1.0,
            gru_lr_ratio: float = 1.0,
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
        self.weight_decay = weight_decay
        self._automatic_optimization = optimization_mode == 'auto'
        self.emb_lr_ratio = emb_lr_ratio
        self.backbone_lr_ratio = backbone_lr_ratio
        self.head_lr_ratio = head_lr_ratio
        self.gru_lr_ratio = gru_lr_ratio
        self.debug = debug
        if debug:
            self.shared_params = list(self.parameters())

    def module_step(self, batch, batch_idx):
        raise NotImplementedError("module_step must be implemented by child class")
        return loss, log_values

    def visualize_results(self, batch, batch_idx):
        return {}

    def on_train_epoch_start(self) -> None:
        self.print(f"Epoch: {self.current_epoch}, LR: {self._optimizer_.param_groups[0]['lr']:1.2e}")
        return super().on_train_epoch_start()
    
    def on_train_epoch_end(self) -> None:
        torch.cuda.empty_cache()
        return super().on_train_epoch_end()

    def on_before_optimizer_step(self, optimizer) -> None:
        if not self.debug:
            return super().on_before_optimizer_step(optimizer)
        # ── 手动梯度裁剪：逐模块独立裁剪，避免 VN-DGCNN 压制其他模块 ──
        # module_clip_val = {
        #     "emb": 100.0,       # VN-DGCNN 梯度大，单独收紧
        #     "backbone": 50.0,   # Cross-Attention
        #     "head": 50.0,       # Flow Head
        #     "gru": 50.0,        # GRU 精调（如有）
        # }

        # pre_clip_norm = {}  # 每个模块裁剪前全局范数
        # post_clip_norm = {}  # 每个模块裁剪后全局范数

        # if getattr(self.model, "get_parameters", None) is not None:
        #     for module_name, max_norm in module_clip_val.items():
        #         params = self.model.get_parameters(module_name)
        #         if not params:
        #             continue

                # 计算裁剪前该模块的范数
                # total_norm = 0.0
                # for p in params:
                #     if p.grad is not None:
                #         total_norm += p.grad.detach().norm(2).item() ** 2
                # total_norm = total_norm ** 0.5
                # pre_clip_norm[f"grad/pre_clip_{module_name}"] = total_norm

                # 对该模块独立裁剪
                # torch.nn.utils.clip_grad.clip_grad_norm_(params, max_norm)

                # # 裁剪后范数
                # total_norm = 0.0
                # for p in params:
                #     if p.grad is not None:
                #         total_norm += p.grad.detach().norm(2).item() ** 2
                # post_clip_norm[f"grad/post_clip_{module_name}"] = total_norm ** 0.5

        # DEBUG: 尝试单独参加project_flow
        # if hasattr(self.model, "head_action"):
        #     params = self.model.head_action.proj_flow.parameters()
        #     torch.nn.utils.clip_grad.clip_grad_norm_(params, 10)
        # if hasattr(self.model, "head_anchor"):
        #     params = self.model.head_anchor.proj_flow.parameters()
        #     torch.nn.utils.clip_grad.clip_grad_norm_(params, 10)
        # if hasattr(self.model, "emb_nn_action"):
        #     params = self.model.emb_nn_action.named_parameters()
        #     for name, param in params:
        #         if not params:
        #             continue
        #         torch.nn.utils.clip_grad.clip_grad_norm_(param, 100)

        # ── 梯度日志记录 ──
        grad_log_period = getattr(self, 'image_log_period', 500)
        if self.global_step < 500:
            return
        if self.global_step % grad_log_period != 0:
            return

        if wandb.run is None:
            return
        wandb_logs = {}
        # wandb_logs = {
        #     "trainer/global_step": self.global_step,
        #     **pre_clip_norm,
        #     **post_clip_norm,
        # }
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                # wandb_logs[f"grad_hist/{name}"] = wandb.Histogram(
                #     grad.cpu().numpy())
                grad_norm = grad.norm(2)
                weight_norm = param.data.norm(2)
                wandb_logs[f"grad_norm/{name}"] = grad_norm.item()
                wandb_logs[f"grad_weight_ratio/{name}"] = (
                    grad_norm / (weight_norm + 1e-8)).item()

        if wandb_logs:
            wandb.log(wandb_logs)

    # def on_before_zero_grad(self, optimizer) -> None:
    #     if getattr(self, "tensorboard_writer", None) is None or \
    #             getattr(self, "_ori_losses", None) is None:
    #         return
    #     grad_list = []
    #     for l in self._ori_losses:
    #         optimizer.zero_grad()
    #         l.backward(retain_graph=True)
    #         grad_A = [p.grad.clone().view(-1) for p in self.shared_params if p.grad is not None]
    #         grad_A = torch.cat(grad_A)   # 展平为一维向量
    #         grad_list.append(grad_A)
    #     # 3. 计算余弦相似度
    #     point_loss_grad, smoothness_loss_grad, dense_loss_grad = grad_list
    #     cos_sim_pc_vpc = torch.dot(point_loss_grad, dense_loss_grad) / (torch.norm(point_loss_grad) * torch.norm(dense_loss_grad) + 1e-8)
    #     cos_sim_pc_sm = torch.dot(point_loss_grad, smoothness_loss_grad) / (torch.norm(point_loss_grad) * torch.norm(smoothness_loss_grad) + 1e-8)
    #     cos_sim_vpc_sm = torch.dot(dense_loss_grad, smoothness_loss_grad) / (torch.norm(dense_loss_grad) * torch.norm(smoothness_loss_grad) + 1e-8)
    #     self.tensorboard_writer.add_scalar("cos_sim_pc_vpc", cos_sim_pc_vpc, self.global_step)
    #     self.tensorboard_writer.add_scalar("cos_sim_pc_sm", cos_sim_pc_sm, self.global_step)
    #     self.tensorboard_writer.add_scalar("cos_sim_pc_vpc_sm", cos_sim_vpc_sm, self.global_step)
    #     return

    def training_step(self, batch, batch_idx):
        self.train()
        loss, log_values = self.module_step(batch, batch_idx)
        log_values.update(lr=self._optimizer_.param_groups[0]["lr"])
        if isinstance(loss, tuple):
            self._ori_losses = loss
            loss = sum(loss)
        for key, val in log_values.items():
            self.log(key, val, logger=True, sync_dist=True)

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
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        loss, log_values = self.module_step(batch, batch_idx)
        if isinstance(loss, tuple):
            loss = sum(loss)
        for key, val in log_values.items():
            self.log("val_" + key, val, logger=True, sync_dist=True)

        if (self.global_val_step % self.image_log_period) == 0 and \
                self.trainer.is_global_zero:
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

        self.log("val_loss", loss, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        self.eval()
        loss, log_values = self.module_step(batch, batch_idx)
        if isinstance(loss, tuple):
            loss = sum(loss)
        for key, val in log_values.items():
            self.log(key, val, logger=True, sync_dist=True)

        if (self.global_step % self.image_log_period) == 0 and \
                self.trainer.is_global_zero:
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

        self.log("test_loss", loss, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # 分层学习率：通过 model.get_parameters(module) 获取各模块参数
        module_lr_map = {
            "emb": self.emb_lr_ratio,
            "backbone": self.backbone_lr_ratio,
            "head": self.head_lr_ratio,
            "gru": self.gru_lr_ratio,
        }
        param_groups = []
        if getattr(self.model, "get_parameters", None) is not None:
            for module_name, ratio in module_lr_map.items():
                params = self.model.get_parameters(module_name)
                if params:
                    param_groups.append({
                        "params": params,
                        "lr": self.lr * ratio,
                        "name": module_name,
                    })
        if not param_groups:
            # fallback：模型未实现 get_parameters，使用所有权重
            param_groups = [{"params": self.parameters()}]

        optimizer = torch.optim.AdamW(
            param_groups, lr=self.lr, weight_decay=self.weight_decay)
        self._optimizer_ = optimizer
        if self.warmup_steps <= 0:
            return optimizer
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        if self.lr_scheduler == 'constant':
            milestones = [self.end_lr_steps]
            # use MilestoneScheduler with warmup rather than constant lr
            self.scheduler = scheduler = MilestoneScheduler(
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
            self.scheduler = scheduler = MilestoneScheduler(
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
            self.scheduler = scheduler = LinearAnnealingWarmup(
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
