# File: rgbir.py
# Description: Interpolate IR pixels with neigboring rgb
# Created: 2023/5/18 17:54
# Author: Samuel Theofie (samueltheofie@gmail.com)

import numpy as np

from .basic_module import BasicModule
from .helpers import get_color_indices, transform_red_to_blue, transform_IR_to_red, subtract_IR

class RGBIR(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.red_cut = self.params.red_cut
        self.green_cut = self.params.green_cut
        self.blue_cut = self.params.blue_cut

    def execute(self, data):
        bayer = data['bayer'].astype(np.int32)
        rgb_ir_indices = {
                'blue'  : ((0, 0), (2, 2)),
                'red'   : ((0, 2), (2, 0)),
                'green' : ((0, 1), (0, 3), (1, 0), (1, 2),
                            (2, 1), (2, 3), (3, 0), (3, 2)),
                'IR'    : ((1, 1), (1, 3), (3, 1), (3, 3)),
                'IR_pos': ((1, 1), (3, 3)),
                'IR_neg': ((1, 3), (3, 1))
                }
    
        IR_color_channel = data['ir']

        IR_pos_indices = get_color_indices('IR_pos', rgb_ir_indices, bayer)
        IR_neg_indices = get_color_indices('IR_neg', rgb_ir_indices, bayer)
        transform_IR_to_red(IR_pos_indices, IR_neg_indices, bayer, rgb_ir_indices)

        red_indices = get_color_indices('red', rgb_ir_indices, bayer)
        transform_red_to_blue(red_indices, bayer, rgb_ir_indices)

        subtract_IR(bayer, IR_color_channel, self.red_cut, self.green_cut, self.blue_cut)

        data['bayer'] = bayer.astype(np.uint16)