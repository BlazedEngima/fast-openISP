# File: irg.py
# Description: Generate IR image
# Created: 2023/5/18 17:54
# Author: Samuel Theofie (samueltheofie@gmail.com)

import numpy as np

from .basic_module import BasicModule
from .helpers import guided_upsampling

class IRG(BasicModule):
    def execute(self, data):
        if 'ir' in data:
            upsampled_ir = guided_upsampling(data['ir'], data['bayer'])
            data['ir'] = upsampled_ir