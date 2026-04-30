from typing import Any
import wandb

import torch
from torch import nn
from torchvision.transforms import ToTensor

from taxpose.training.flow_equivariance_training_module_nocentering import EquivarianceTrainingModule

from taxpose.utils.se3 import mse_criterion

from LibMTL.weighting import CAGrad
from LibMTL.architecture import AbsArchitecture

import matplotlib.cm as cm
import numpy as np


to_tensor = ToTensor()


class EquivarianceTrainingModuleCaGrad(EquivarianceTrainingModule):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super(EquivarianceTrainingModuleCaGrad, self).__init__(*args, **kwargs)

        class MTLmodel(AbsArchitecture, CAGrad):
            def __init__(self, encoder, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.encoder = encoder
                self.init_param()
                
        self.cagrad_weighting = MTLmodel(
            encoder=self.model,
            task_name=['point', 'dense', 'smooth'],
            encoder_class='Custom',
            decoders=None,
            rep_grad=False,
            multi_input=True,
            device='cuda',
            kwargs={})

        # # self.cagrad_weighting = CAGrad() # 初始化CAGrad
        # if not hasattr(self.cagrad_weighting, 'rep_grad'):
        #     self.cagrad_weighting.rep_grad = False
        # else:
        #     self.cagrad_weighting.rep_grad = False

    # def training_step(self, batch, batch_idx):
    #     loss, log_values = self.module_step(batch, batch_idx)
    #     if isinstance(loss, tuple):
    #         self._ori_losses = loss
    #         loss = sum(loss)
    #     for key, val in log_values.items():
    #         self.log(key, val, logger=True)

    #     if (self.global_step % self.image_log_period) == 0:
    #         results_images = self.visualize_results(batch, batch_idx)

    #         for key, val in results_images.items():
    #             if isinstance(val, wandb.Object3D) and wandb.run is not None:
    #                 wandb.log(
    #                     {
    #                         key: val,
    #                         "trainer/global_step": self.global_step,
    #                     }
    #                 )
    #             elif self.logger is not None:
    #                 self.logger.log_image(
    #                     key,
    #                     images=[val],  # self.global_step
    #                 )
    #     self.log("train_loss", loss, prog_bar=True, logger=True)

    #     # self._optimizer_.zero_grad()
    #     # 2. 使用LibMTL提供的backward方法
    #     # 它的内部会自动处理梯度收集和CAGrad计算
    #     # self.cagrad_weighting.backward(
    #     #     losses=self._ori_losses,
    #     #     calpha=0.5,
    #     #     rescale=1.0,
    #     #     parameters=list(self.parameters()), # 传入要优化的参数
    #     #     # 可以选择性地传入各任务的计算图，但rep_grad必须为False
    #     #     # rep_grad=False 
    #     # )
    #     # self._optimizer_.step()
    #     # self.lr_schedulers().step()
    #     return loss  # PL需要，但梯度已自行处理

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        assert self.automatic_optimization
        self.cagrad_weighting.backward(
            losses=self._ori_losses,
            calpha=0.5,
            rescale=1.0,
            parameters=list(self.parameters()), # 传入要优化的参数
            # 可以选择性地传入各任务的计算图，但rep_grad必须为False
            # rep_grad=False 
        )
