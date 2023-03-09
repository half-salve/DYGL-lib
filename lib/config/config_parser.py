"""
Basic DG config module
"""

import os
import json
import torch

from .util import get_config_dir
class DGCONFIG(object):
    r"""The basic DGCONFIG for generating a list of hyperparameters,.
    This class defines a basic template class for CONFIG.
    The following steps will are executed automatically:

      1. Check whether args is NONE. If true, goto 3.
      2. Call ``parse_config_file()`` If the default model parameters are used, the function will read the parameters according to the specified model.
      3. Call ``parse_external_config()`` Personalize custom arguments based on args and other_args.
      4. Call ``init_device()`` Initialize the device.
      6. Done.
          """
    def __init__(self, args,config_file=None):
        self._args=vars(args)
        self._config_dir=None
        self._config_file = config_file
        self._config = {}

        if self.config_file is None:
            self._config_dir = get_config_dir()

        #通过函数的形式对config进行添加补全
        self._parse_config_file()
        self._parse_external_config()
        self._init_device()

    def _parse_config_file(self):

        if self.config_file is None:
            # TODO: 对 config file 的格式进行检查
            self.Check_required_parameters()

            self.read_defalt_parameters()

        else:
            self.read_defalt_parameters()


    def _parse_external_config(self):
        if self.args is not None :
            self.external_config()

    def _init_device(self):
        pass

    def Check_required_parameters(self):
        pass

    def read_defalt_parameters(self):
        pass

    def external_config(self):
        pass

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config

    # 支持迭代操作
    def __iter__(self):
        return self.config.__iter__()

    @property
    def config(self):
        r"""return config.
        """
        return self._config

    @property
    def config_dir(self):
        r"""return config.
        """
        return self._config_dir

    @property
    def args(self):
        r"""return args.
               """
        return self._args
        r"""return other_args.
               """
        return self._other_args

    @property
    def config_file(self):
        r"""return other_args.
               """
        return self._config_file


class ConfigParser(DGCONFIG):
    """
    use to parse the user defined parameters and use these to modify the
    pipeline's parameter setting.
    值得注意的是，目前各阶段的参数是放置于同一个 dict 中的，因此需要编程时保证命名空间不冲突。
    config 优先级：命令行 > config file 
    """

    def __init__(self, args,config_file=None,other_args=None):
        """
        Args:
            task, model, dataset (str): 用户在命令行必须指明的三个参数
            config_file (str): 配置文件的文件名，将在项目根目录下进行搜索
            other_args (dict): 通过命令行传入的其他参数
        """
        super(ConfigParser, self).__init__(args = args,
                                       config_file=config_file ,
                                )

    def Check_required_parameters(self):

        if self.args.get("model",None) is None:
            raise ValueError('the parameter <model> should not be None!')

        if self.args.get("dataset",None) is None:
            raise ValueError('the parameter <dataset> should not be None!')

    def read_defalt_parameters(self):

        self._config_file = os.path.join(self.config_dir,"{}.json".format(self.args["model"]))

        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                x = json.load(f)
                for key in x:
                    if key not in self.config:
                        self.config[key] = x[key]
        else:
            raise FileNotFoundError(
                'Config file {} is not found. Please ensure \
                the config file is in the root dir and is a JSON \
                file.'.format(self.config_file))

    def external_config(self):
        for key in self.args:
            if key not in self.config:
                self.config[key] = self.args[key]
            else :
                if self.config[key]!=self.args[key]:
                    print("The value of {} in the default config is {} <type:{}>,which will now be replaced by {} <type:{}> in args"\
                        .format(key,self.config[key],type(self.config[key]),self.args[key],type(self.args[key])))
                    self.config[key] = self.args[key]


    def _init_device(self):
        gpu_id = self.config.get('gpu', -1)
        self.config['device'] = torch.device(
            "cuda:%d" % gpu_id if gpu_id>-1  else "cpu")

