import collections
import copy


class BaseModuleBuilder():
    """
    初始化一个有序字典
    OrderedDict会根据放入元素的先后顺序进行排序
    """
    REGISTERED_MODULES = collections.OrderedDict()

    def __init__(self, requires_register_modules=None, requires_renew_modules=None):
        if requires_register_modules is not None and isinstance(requires_register_modules,
                                                                (dict, collections.OrderedDict)):
            for name, module in requires_register_modules.items():
                self.register(name, module)
        if requires_renew_modules is not None and isinstance(requires_renew_modules, (dict, collections.OrderedDict)):
            for name, module in requires_renew_modules.items():
                self.renew(name, module)
        self.validate()

    def build(self, module_cfg):
        module_cfg = copy.deepcopy(module_cfg)
        module_type = module_cfg.pop('type')
        module = self.REGISTERED_MODULES[module_type](**module_cfg)
        return module

    def register(self, name, module):
        assert callable(module)
        assert name not in self.REGISTERED_MODULES
        self.REGISTERED_MODULES[name] = module

    def renew(self, name, module):
        assert callable(module)
        assert name in self.REGISTERED_MODULES
        self.REGISTERED_MODULES[name] = module

    def validate(self):
        for _, module in self.REGISTERED_MODULES.items():
            assert callable(module)

    def delete(self, name):
        assert name in self.REGISTERED_MODULES
        del self.REGISTERED_MODULES[name]

    def pop(self, name):
        assert name in self.REGISTERED_MODULES
        module = self.REGISTERED_MODULES.pop(name)
        return module

    def get(self, name):
        assert name in self.REGISTERED_MODULES
        module = self.REGISTERED_MODULES.get(name)
        return module

    def items(self):
        return self.REGISTERED_MODULES.items()

    def clear(self):
        return self.REGISTERED_MODULES.clear()

    def values(self):
        return self.REGISTERED_MODULES.values()

    def keys(self):
        return self.REGISTERED_MODULES.keys()

    def copy(self):
        return self.REGISTERED_MODULES.copy()

    def update(self, requires_update_modules):
        return self.REGISTERED_MODULES.update(requires_update_modules)
