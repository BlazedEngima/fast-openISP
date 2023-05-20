import os
import os.path as op

import cv2
import numpy as np
import skimage.io

from pipeline import Pipeline
from utils.yacs import Config


OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

    
def pre_process(self, bayer, show_intermediate=False, verbose=True):
    """
    ISP pipeline execution
    :param bayer: input Bayer array, np.ndarray(H, W)
    :param save_intermediates: whether to save intermediate results from all ISP modules
    :param verbose: whether to print timing messages
    :return:
        data: a dict containing results from different domains (Bayer, RGB, and YCbCr)
            and the final RGB output (data['output'])
        intermediates: a dict containing intermediate results if save_intermediates=True,
            otherwise a empty dict
    """
    

    
    pass

def demo_rgb_ir_raw():
    cfg = Config('configs/RGB_IR.yaml')
    pipeline = Pipeline(cfg)

    raw_path = 'raw/Bright_color.raw'
    bayer = np.fromfile(raw_path, dtype='uint16', sep='')
    bayer = bayer.reshape((cfg.hardware.raw_height, cfg.hardware.raw_width))
    # bayer = pre_process(bayer)
    data, _ = pipeline.execute(bayer)

    output_path = op.join(OUTPUT_DIR, 'rbg_ir_test_matrix_with_blc.png')

    # np.savetxt(output_path, bayer)
    output = cv2.cvtColor(data['output'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output)

if __name__ == '__main__':
    print('Processing test raw...')
    demo_rgb_ir_raw()