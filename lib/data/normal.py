import numpy as np
import torch

import os

from .dynamic_dataset  import DGBuiltinDataset

class normalDataset(DGBuiltinDataset):
    def __init__(self, name, raw_dir=None, force_reload=False, verbose=False):
        assert raw_dir, 'raw_dir not detected please re-enter'
        super(normalDataset, self).__init__(name=name,
                                            url=None,
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)


