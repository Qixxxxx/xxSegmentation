import torch.nn as nn

from utils import BaseModuleBuilder
from .hardsigmoid import HardSigmoid
from .hardswish import HardSwish
from .swish import Swish


class ActivationBuilder(BaseModuleBuilder):
    # 模块字典
    REGISTERED_MODULES = {
        'ReLU': nn.ReLU,
        'GELU': nn.GELU,
        'ReLU6': nn.ReLU6,
        'PReLU': nn.PReLU,
        'Sigmoid': nn.Sigmoid,
        'LeakyReLU': nn.LeakyReLU,
        'HardSwish': HardSwish,
        'HardSigmoid': HardSigmoid,
        'Swish': Swish,
    }
    # 初始化时在torch.nn中寻找对应的激活函数，找到则放入模块字典中
    for act_type in ['ELU', 'Hardshrink', 'Hardtanh', 'LogSigmoid', 'RReLU', 'SELU', 'CELU', 'SiLU', 'GLU',
                     'Mish', 'Softplus', 'Softshrink', 'Softsign', 'Tanh', 'Tanhshrink', 'Threshold']:
        if hasattr(nn, act_type):
            REGISTERED_MODULES[act_type] = getattr(nn, act_type)

    def build(self, act_cfg):
        if act_cfg is None:
            return nn.Identity()
        return super().build(act_cfg)


BuildActivation = ActivationBuilder().build
