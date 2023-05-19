from collections import OrderedDict
import numpy as np
import os
import math
import time
import skimage.filters as filters
import skimage.transform as transform
import cv2
import modules.helpers

OUTPUT_DIR = './output'

raw_path = 'raw/Bright_color.raw'
bayer = np.fromfile(raw_path, dtype='uint16', sep='')
bayer = bayer.reshape((1944, 2592))

# ordered_dict = OrderedDict(bayer=test_array)
# kernel = np.array([ 
#                         [ 'B','G','R','G'   ],
#                         [ 'G','IR','G','IR' ],
#                         [ 'R','G','B','G'   ],
#                         [ 'G','IR','G','IR' ]
#                      ])

# bayer = np.array([ 
#                         [  1,  2,  3, 4 ],
#                         [  5,  6,  7, 8 ],
#                         [  9, 10, 11, 12],
#                         [ 13, 14, 15, 16]
#                      ])

rgb_ir_indices = {
                    'blue'  : ((0, 0), (2, 2)),
                    'red'   : ((0, 2), (2, 0)),
                    'green' : ((0, 1), (0, 3), (1, 0), (1, 2),
                               (2, 1), (2, 3), (3, 0), (3, 2)),
                    'IR'    : ((1, 1), (1, 3), (3, 1), (3, 3))
                 }

sub_arrays = {}

for color, indices in rgb_ir_indices.items():
    start_time = time.time()
    small_mask = np.zeros((4, 4), dtype=bool)
    for idx in indices:
        small_mask[idx] = True

    repetitions = (
                    math.ceil(bayer.shape[0] / small_mask.shape[0]), 
                    math.ceil(bayer.shape[1] / small_mask.shape[1])
                   )

    mask = np.tile(small_mask, repetitions)

    color_array = np.where(mask, bayer, np.nan)
    sub_arrays[color] = color_array
    end_time = time.time() - start_time
    # print(f'{color}_array done. Time elapsed {end_time:.3f}s')
    # np.savetxt(output_path, color_array, fmt='%d')

def interpolate_vertical(color, coords):
    color_array = sub_arrays[color]
    interpolation_axis = set(coords[i][0] for i in range(len(coords)))
    interpolation_axis = list(interpolation_axis)
    y = np.arange(color_array.shape[0])

    col_mask = np.zeros((color_array.shape[0], 4), dtype=bool)
    col_mask[:, interpolation_axis] = True

    repetitions = (1, math.ceil(color_array.shape[1] / col_mask.shape[1]))

    mask = np.tile(col_mask, repetitions)
    interpolated_matrix = np.zeros_like(color_array)
    
    for col in range(color_array.shape[1]):
        x = color_array[:, col]
        interpolated_col = np.interp(y, y[~np.isnan(x)], x[~np.isnan(x)])
        interpolated_matrix[:, col] = interpolated_col

    interpolated_color_array = np.where(mask, interpolated_matrix, np.nan)
    return interpolated_color_array

def interpolate_horizontal(color, coords):
    color_array = sub_arrays[color]
    interpolation_axis = set(coords[i][0] for i in range(len(coords)))
    interpolation_axis = list(interpolation_axis)
    # interpolation_axis = [coords[0][0], coords[1][0]]
    x = np.arange(color_array.shape[1])

    row_mask = np.zeros((4, color_array.shape[1]), dtype=bool)
    row_mask[interpolation_axis] = True

    repetitions = (math.ceil(color_array.shape[0] / row_mask.shape[0]), 1)

    mask = np.tile(row_mask, repetitions)
    
    interpolated_matrix = np.zeros_like(color_array)
    
    for row in range(color_array.shape[0]):
        y = color_array[row, :]
        interpolated_row = np.interp(x, x[~np.isnan(y)], y[~np.isnan(y)])
        interpolated_matrix[row, :] = interpolated_row

    interpolated_color_array = np.where(mask, interpolated_matrix, np.nan)
    return interpolated_color_array


np.set_printoptions(
    suppress=True, 
    formatter={
        'int_kind':'{:d}'.format
    }
)

def guided_upsampling(image, guide, radius=5, epsilon=0.01):
    # Compute guidance image from the low-resolution image
    print(guide)
    guidance_image = cv2.GaussianBlur(guide, (0, 0), 1.0)
    print(guidance_image)
    # Compute details or high-frequency information
    # details = image_lr - guidance_image

    # # Upsample the low-resolution image
    # image_upsampled = transform.resize(image_lr, image_hr.shape, mode='reflect')

    # # Apply guided filter to refine the upsampled image
    # upsampled_filtered = filters.guided_filter(guidance_image, image_upsampled, radius, epsilon)

    # # Add the details back to the upsampled image
    # image_hr_upsampled = upsampled_filtered + details

    # return image_hr_upsampled

print(guided_upsampling(rgb_ir_indices['red'], interpolate_horizontal('green', rgb_ir_indices['red'])))