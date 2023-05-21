# File: ir_cut.py
# Description: Interpolate IR pixels with neigboring rgb
# Created: 2023/5/18 17:54
# Author: Samuel Theofie (samueltheofie@gmail.com)

import numpy as np

from .basic_module import BasicModule
from .helpers import extract_IR, get_color_indices, transform_red_to_blue, transform_IR_to_red, fill_zeroes, subtract_IR

class IRC(BasicModule):
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
    
        IR_color_channel = extract_IR(bayer, rgb_ir_indices)
        filled_IR_color_channel = fill_zeroes(IR_color_channel, (2, 2), (1, 1))

        data['IR'] = IR_color_channel
        # save_to_txt('IR_image_channel.txt', IR_color_channel)

        # IR_indices = get_color_indices('IR', rgb_ir_indices, bayer)
        IR_pos_indices = get_color_indices('IR_pos', rgb_ir_indices, bayer)
        IR_neg_indices = get_color_indices('IR_neg', rgb_ir_indices, bayer)
        transform_IR_to_red(IR_pos_indices, IR_neg_indices, bayer, rgb_ir_indices)

        red_indices = get_color_indices('red', rgb_ir_indices, bayer)
        transform_red_to_blue(red_indices, bayer, rgb_ir_indices)

        # save_to_txt('averaged_red_values.txt', changed_red_to_blue)

        # save_to_txt('averaged_IR_values.txt', changed_IR_to_red)

        subtract_IR(bayer, filled_IR_color_channel)

        data['bayer'] = bayer.astype(np.uint16)