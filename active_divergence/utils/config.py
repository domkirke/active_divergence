import yaml, copy, argparse

class ConfigItem(object):
    def __getattr__(self, item):
        if not item in self._config_items:
            return None
        return self.__dict__[item]

    def __setattr__(self, key, value, record=True):
        self.__dict__[key] = value
        if record and (not key in self._config_items):
            self._config_items.append(key)

    def __repr__(self):
        return "("+", ".join(["%s: %s"%(s, self.__getattribute__(s)) for s in self._config_items])+")"

    def __contains__(self, item):
        return item in self._config_items

    def __init__(self, obj={}, **kwargs):
        self.__setattr__('_config_items', [], False)
        if isinstance(obj, argparse.Namespace):
            keys = list(filter(lambda x: x[0] != "_", dir(obj)))
            obj = {k: getattr(obj, k) for k in keys}
        obj = {**obj, **kwargs}
        for k, v in obj.items():
            if isinstance(v, dict):
                self.__setattr__(k, ConfigItem(v))
            else:
                self.__setattr__(k, v)

    def keys(self):
        return self._config_items

    def __getitem__(self, item):
        if not item in self._config_items:
            raise KeyError()
        return self.__getattr__(item)

    def __add__(self, obj):
        if not isinstance(obj, ConfigItem):
            raise TypeError("ConfigItem can only be added to an other ConfigItem")
        concat_dict = {**dict(self), **dict(obj)}
        return ConfigItem(concat_dict)

    def dict(self):
        current_dict = {}
        for k in self._config_items:
            if isinstance(self.__getattr__(k), ConfigItem):
                current_dict[k] = self.__getattr__(k).dict()
            else:
                current_dict[k] = self.__getattr__(k)
        return current_dict


    def clone(self):
        conf = type(self)()
        for k in self._config_items:
            if isinstance(self.__getattr__(k), ConfigItem):
                conf.__setattr__(k, self.__getattr__(k).clone())
            else:
                conf.__setattr__(k, copy.deepcopy(self.__getattr__(k)))
        return conf

    def write(self, path):
        with open(path, "w+") as f:
            f.write(yaml.dump(self.dict()))

class Config(ConfigItem):
    def __repr__(self):
        return "CONFIG"+super(Config, self).__repr__()

    def __init__(self, obj={}, **kwargs):
        """
        Creates a configs object from a yaml file, or dict
        Args:
            obj (str or dict): if str, loads corresponding YAML file. If dict, parses dict into a new Config object.
        """
        if isinstance(obj, str):
            with open(obj) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            super(Config, self).__init__(config)
        elif isinstance(obj, ConfigItem):
            super(Config, self).__init__(obj.dict())
        elif isinstance(obj, argparse.Namespace):
            keys = list(filter(lambda x: x[0] != "_", dir(obj)))
            obj = {k: getattr(obj, k) for k in keys}
            super(Config, self.__init__(obj))
        else:
            obj = {**obj, **kwargs}
            super(Config, self).__init__(obj)

