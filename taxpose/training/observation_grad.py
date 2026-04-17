import torch
import torch.nn as nn


def get_layer_grad_norms(model: nn.Module):
    """获取模型每一层的梯度范数"""
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 计算 L2 范数（最常用）
            norm = param.grad.norm(2).item()
            grad_norms[name] = norm
    return grad_norms