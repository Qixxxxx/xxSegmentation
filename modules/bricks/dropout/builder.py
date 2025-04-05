import torch.nn as nn

from .droppath import DropPath
from utils import BaseModuleBuilder


class DropoutBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'DropPath': DropPath, 'Dropout': nn.Dropout, 'Dropout2d': nn.Dropout2d, 'Dropout3d': nn.Dropout3d,
    }

    def build(self, dropout_cfg):
        if dropout_cfg is None:
            return nn.Identity()
        return super().build(dropout_cfg)


BuildDropout = DropoutBuilder().build
