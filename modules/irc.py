# File: irc.py
# Description: Extract the IR pixels from rgb_ir and apply a clip
# Created: 2023/5/18 17:54
# Author: Samuel Theofie (samueltheofie@gmail.com)

import numpy as np

from .basic_module import BasicModule
from .helpers import subtract_IR

class IRC(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.clip_lvl = self.params.clip_lvl

    def execute(self, data):
        bayer = data['bayer'].astype(np.int32)
        ir = bayer[1::2, 1::2]
        
        ir = np.clip(ir, 0, np.max(bayer // self.clip_lvl))

        data['ir'] = ir
