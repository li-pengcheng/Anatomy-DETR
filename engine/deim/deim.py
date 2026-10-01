"""
Anatomy-DETR: tooth-aware DETR for dental lesion detection.
Built upon DEIM (DETR with Improved Matching for Fast Convergence).
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch.nn as nn
from ..core import register


__all__ = ['AnatomyDETR', 'DEIM']


@register()
@register(name='DEIM')
class AnatomyDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def set_epoch(self, epoch):
        if hasattr(self.decoder, 'set_epoch'):
            self.decoder.set_epoch(epoch)

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


# Backward-compatible alias for existing DEIM configs/checkpoints.
DEIM = AnatomyDETR
