import os
import os.path as op

import cv2
import numpy as np

from pipeline import Pipeline
from utils.yacs import Config


OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def demo_rgb_ir_raw_dim():
    cfg = Config('configs/RGB_IR_dim.yaml')
    pipeline = Pipeline(cfg)

    raw_path = 'raw/Dim_color.raw'
    bayer = np.fromfile(raw_path, dtype='uint16', sep='')
    bayer = bayer.reshape((cfg.hardware.raw_height, cfg.hardware.raw_width))
    data, _ = pipeline.execute(bayer)

    output_path = op.join(OUTPUT_DIR, 'rbg_ir_test_dim.png')
    output_path_ir = op.join(OUTPUT_DIR, 'rbg_ir_test_dim_ir.png')

    output = cv2.cvtColor(data['output'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output)

    if 'ir' in data:
        output_ir = data['ir']
        cv2.imwrite(output_path_ir, output_ir)

def demo_rgb_ir_raw_bright():
    cfg = Config('configs/RGB_IR_bright.yaml')
    pipeline = Pipeline(cfg)

    raw_path = 'raw/Bright_color.raw'
    bayer = np.fromfile(raw_path, dtype='uint16', sep='')
    bayer = bayer.reshape((cfg.hardware.raw_height, cfg.hardware.raw_width))
    data, _ = pipeline.execute(bayer)

    output_path = op.join(OUTPUT_DIR, 'rbg_ir_test_bright.png')
    output_path_ir = op.join(OUTPUT_DIR, 'rbg_ir_test_bright_ir.png')

    output = cv2.cvtColor(data['output'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output)

    if 'ir' in data:
        output_ir = data['ir']
        cv2.imwrite(output_path_ir, output_ir)

if __name__ == '__main__':
    print('Processing test raw dim...')
    demo_rgb_ir_raw_dim()
    print('Processing test raw bright...')
    demo_rgb_ir_raw_bright()