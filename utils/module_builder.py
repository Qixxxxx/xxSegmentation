import copy
import collections


class BaseModuleBuilder():
    """
    初始化一个有序字典
    OrderedDict会根据放入元素的先后顺序进行排序
    """
    REGISTERED_MODULES = collections.OrderedDict()

    def __init__(self, requires_register_modules=None, requires_renew_modules=None):
        if requires_register_modules is not None and isinstance(requires_register_modules, (dict, collections.OrderedDict)):
            for name, module in requires_register_modules.items():
                self.register(name, module)
        if requires_renew_modules is not None and isinstance(requires_renew_modules, (dict, collections.OrderedDict)):
            for name, module in requires_renew_modules.items():
                self.renew(name, module)
        self.validate()

    '''build'''
    def build(self, module_cfg):
        module_cfg = copy.deepcopy(module_cfg)
        module_type = module_cfg.pop('type')
        module = self.REGISTERED_MODULES[module_type](**module_cfg)
        return module

    '''register'''
    def register(self, name, module):
        assert callable(module)
        assert name not in self.REGISTERED_MODULES
        self.REGISTERED_MODULES[name] = module

    '''renew'''
    def renew(self, name, module):
        assert callable(module)
        assert name in self.REGISTERED_MODULES
        self.REGISTERED_MODULES[name] = module

    '''validate'''
    def validate(self):
        for _, module in self.REGISTERED_MODULES.items():
            assert callable(module)