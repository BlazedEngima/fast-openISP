import numpy as np
import math
import os
import time

np.set_printoptions(
    suppress=True, 
    formatter={
        'int_kind':'{:d}'.format
    }
)

OUTPUT_DIR = './output'

def save_to_txt(name, obj):
    output_path = os.path.join(OUTPUT_DIR, name)
    np.savetxt(output_path, obj, fmt='%d')
    
def mask_by_color(color, pattern_dict, image):
    
    # Get indices of the r, g, b, and ir of the 4x4 kernel from the sensor
    indices = pattern_dict[color]

    # Create a mask for the kernel
    small_mask = np.zeros((4, 4), dtype=bool)
    for idx in indices:
        small_mask[idx] = True

    # Calculate how many repetitions to perform to tile the mask for both the height and width
    repetitions = (
                    math.ceil(image.shape[0] / small_mask.shape[0]), 
                    math.ceil(image.shape[1] / small_mask.shape[1])
                   )

    # Extend the mask to make it the full raw resolution
    mask = np.tile(small_mask, repetitions)

    return mask

def get_color_indices(color, pattern_dict, image):
    mask = mask_by_color(color, pattern_dict, image)
    return np.column_stack(np.where(mask))

def extract_IR(image, pattern_dict):
    mask = mask_by_color('IR', pattern_dict, image)
    IR_values = np.where(mask, image, 0)

    return IR_values
    
def transform_red_to_blue(red_index, image, pattern_dict, __mask__=True):
    result = image.copy()
    x = red_index[:, 0]
    y = red_index[:, 1]

    # Initialize zero arrays
    left_blue = np.zeros_like(x)
    top_blue = np.zeros_like(x)
    bottom_blue = np.zeros_like(x)
    right_blue = np.zeros_like(x)

    # Get valid indices within the boundaries of the image indices
    valid_indices = (y - 2 >= 0)
    left_blue[valid_indices] = image[x[valid_indices], y[valid_indices] - 2]

    valid_indices = (x - 2 >= 0)
    top_blue[valid_indices] = image[x[valid_indices] - 2, y[valid_indices]]
    
    valid_indices = (x + 2 < image.shape[0])
    bottom_blue[valid_indices] = image[x[valid_indices] + 2, y[valid_indices]]

    valid_indices = (y + 2 < image.shape[1])
    right_blue[valid_indices] = image[x[valid_indices], y[valid_indices] + 2]

    # Add the surrounding blue pixels together
    numerator = left_blue + right_blue + top_blue + bottom_blue

    # Get the divisor in case if one of them has a 0 for proper averages
    divisor = ((left_blue != 0).astype(int) + 
            (right_blue != 0).astype(int) + 
            (top_blue != 0).astype(int) + 
            (bottom_blue != 0).astype(int))
    
    # Make sure division by 0 doesnt occur
    divisor[divisor == 0] = 1

    # Get the average
    result[x, y] = numerator // divisor

    # Performing masking if wanted
    if __mask__:
        mask = mask_by_color('red', pattern_dict, result)
        result = np.where(mask, result, 0)

    return result

def transform_IR_to_red(IR_pos_index, IR_neg_index, image, pattern_dict, __mask__=True):
    result = image.copy()
    pos_x = IR_pos_index[:, 0]
    pos_y = IR_pos_index[:, 1]
    neg_x = IR_neg_index[:, 0]
    neg_y = IR_neg_index[:, 1]

    top_left_val = np.zeros_like(neg_x)
    top_right_val = np.zeros_like(pos_x)
    bottom_left_val = np.zeros_like(pos_x)
    bottom_right_val = np.zeros_like(neg_x)

    valid_indices = (pos_x + 1 < image.shape[0]) & (pos_y - 1 >= 0)
    top_right_val[valid_indices] = image[pos_x[valid_indices] + 1, pos_y[valid_indices] - 1]

    valid_indices = (pos_x - 1 >= 0) & (pos_y + 1 < image.shape[1])
    bottom_left_val[valid_indices] = image[pos_x[valid_indices] - 1, pos_y[valid_indices] + 1]

    valid_indices = (neg_x - 1 >= 0) & (neg_y - 1 >= 0)
    top_left_val[valid_indices] = image[neg_x[valid_indices] - 1, neg_y[valid_indices] - 1]

    valid_indices = (neg_x + 1 < image.shape[0]) & (neg_y + 1 < image.shape[1])
    bottom_right_val[valid_indices] = image[neg_x[valid_indices] + 1, neg_y[valid_indices] + 1]

    numerator = top_left_val + bottom_right_val
    divisor = ((top_left_val != 0).astype(int) + (bottom_right_val != 0).astype(int))
    divisor[divisor == 0] = 1

    answer = numerator // divisor
    answer[answer == 0] = result[-1, -1]
    result[neg_x, neg_y] = answer

    numerator = top_right_val + bottom_left_val
    divisor = ((top_right_val != 0).astype(int) + (bottom_left_val != 0).astype(int))
    divisor[divisor == 0] = 1

    answer = numerator // divisor
    answer[answer == 0] = result[-1, -1]
    result[pos_x, pos_y] = answer

    if __mask__:
        mask = mask_by_color('IR', pattern_dict, result)
        result = np.where(mask, result, 0)
    
    return result

if __name__ == '__main__':
    raw_path = 'raw/Bright_color.raw'
    bayer = np.fromfile(raw_path, dtype='uint16', sep='')
    bayer = bayer.reshape((1944, 2592))
    sub_arrays = {}
    rgb_ir_indices = {
                    'blue'  : ((0, 0), (2, 2)),
                    'red'   : ((0, 2), (2, 0)),
                    'green' : ((0, 1), (0, 3), (1, 0), (1, 2),
                               (2, 1), (2, 3), (3, 0), (3, 2)),
                    'IR'    : ((1, 1), (1, 3), (3, 1), (3, 3)),
                    'IR_pos': ((1, 1), (3, 3)),
                    'IR_neg': ((1, 3), (3, 1))
                 }
    # save_to_txt('bayer_image.txt', bayer)

    start_time = time.time()
    IR_color_channel = extract_IR(bayer, rgb_ir_indices)

    # save_to_txt('IR_image_channel.txt', IR_color_channel)

    red_indices = get_color_indices('red', rgb_ir_indices, bayer)
    changed_red_to_blue = transform_red_to_blue(red_indices, bayer, rgb_ir_indices)

    # save_to_txt('averaged_red_values.txt', changed_red_to_blue)

    IR_pos_indices = get_color_indices('IR_pos', rgb_ir_indices, bayer)
    IR_neg_indices = get_color_indices('IR_neg', rgb_ir_indices, bayer)
    changed_IR_to_red = transform_IR_to_red(IR_pos_indices, IR_neg_indices, bayer, rgb_ir_indices)

    # save_to_txt('averaged_IR_values.txt', changed_IR_to_red)

    end_time = time.time() - start_time

    print(f'Time elapsed {end_time:.3f}')
    