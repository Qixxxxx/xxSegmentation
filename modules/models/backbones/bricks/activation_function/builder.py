import torch.nn as nn

from modules.models.backbones.bricks.activation_function.hardsigmoid import HardSigmoid
from modules.models.backbones.bricks.activation_function.hardswish import HardSwish
from modules.models.backbones.bricks.activation_function.swish import Swish
from utils import BaseModuleBuilder


class ActivationBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'ReLU': nn.ReLU, 'GELU': nn.GELU, 'ReLU6': nn.ReLU6, 'PReLU': nn.PReLU,
        'Sigmoid': nn.Sigmoid, 'HardSwish': HardSwish, 'LeakyReLU': nn.LeakyReLU,
        'HardSigmoid': HardSigmoid, 'Swish': Swish,
    }
    for act_type in ['ELU', 'Hardshrink', 'Hardtanh', 'LogSigmoid', 'RReLU', 'SELU', 'CELU', 'SiLU', 'GLU',
                     'Mish', 'Softplus', 'Softshrink', 'Softsign', 'Tanh', 'Tanhshrink', 'Threshold']:
        if hasattr(nn, act_type):
            REGISTERED_MODULES[act_type] = getattr(nn, act_type)

    def build(self, act_cfg):
        if act_cfg is None: return nn.Identity()
        return super().build(act_cfg)


'''
BuildActivation
方法指向
'''
BuildActivation = ActivationBuilder().build

if __name__ == '__main__':
    # 两种初始化，一种使用默认初始化，一种使用指定参数初始化
    a = BuildActivation({'type': 'ReLU', 'inplace': True})
    print(a)
    print(type(a))
    b = ActivationBuilder(None, {'ReLU': nn.GELU}).build({'type': 'ReLU'})
    print(b)
    print(type(b))
