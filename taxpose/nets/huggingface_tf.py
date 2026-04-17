from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Transformer(nn.Transformer):
    def __init__(self, d_model: int = 512, nhead: int = 8, 
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, 
                 dim_feedforward: int = 2048, dropout: float = 0.1, 
                 activation: str | Callable[[Tensor], Tensor] = F.relu, 
                 custom_encoder: Any | None = None, custom_decoder: Any | None = None, 
                 norm_first: bool = False, bias: bool = True) -> None:
        super().__init__(
            d_model, nhead, num_encoder_layers, num_decoder_layers, 
            dim_feedforward, dropout, activation, custom_encoder, 
            custom_decoder, batch_first=True, norm_first=norm_first, 
            bias=bias
        )