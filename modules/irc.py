# File: ir_cut.py
# Description: Interpolate IR pixels with neigboring rgb
# Created: 2023/5/18 17:54
# Author: Samuel Theofie (samueltheofie@gmail.com)

import numpy as np

from .basic_module import BasicModule

class IRC(BasicModule):
    def execute(self, data):
        bayer = data['bayer'].astype(np.int32)
        ir = bayer[1::2, 1::2]

        data['ir'] = ir