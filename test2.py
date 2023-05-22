import numpy as np
import math
import os
import time
from modules.helpers import get_bayer_indices
import cv2

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
    
def mask_by_color(indices, image, kernel_size=(4, 4)):
    
    # Create a mask for the kernel
    small_mask = np.zeros(kernel_size, dtype=bool)
    small_mask[indices] = True
    # Calculate how many repetitions to perform to tile the mask for both the height and width
    repetitions = (
                    math.ceil(image.shape[0] / small_mask.shape[0]), 
                    math.ceil(image.shape[1] / small_mask.shape[1])
                   )

    # Extend the mask to make it the full raw resolution
    mask = np.tile(small_mask, repetitions)

    return mask

def get_color_indices(color, pattern_dict, image, kernel_size=(4, 4)):
    indices = pattern_dict[color]
    mask = mask_by_color(indices, image, kernel_size)
    return np.column_stack(np.where(mask))

def extract_IR(image, pattern_dict, kernel_size=(4, 4)):
    indices = pattern_dict['IR']
    mask = mask_by_color(indices, image, kernel_size)
    IR_values = np.where(mask, image, 0)

    return IR_values

def extract_IR_2(image, pattern_dict, kernel_size=(4, 4)):
    indices = pattern_dict['IR']
    mask = mask_by_color(indices, image, kernel_size)
    IR_values = np.where(mask, image, 0)
    IR_values = IR_values[np.nonzero(IR_values)]
    new_size =  (
                    math.ceil(image.shape[0] // 2),
                    math.ceil(image.shape[1] // 2),
                )

    IR_values = IR_values.reshape(new_size)

    return IR_values

def transform_red_to_blue(red_index, image, pattern_dict, __mask__=False):
    x = red_index[:, 0]
    y = red_index[:, 1]

    # Pad original image with 2 rows and columns of zero on the right and bottom side
    padded_image = np.pad(image, (0, 2), mode='constant')

    # Get valid indices within the boundaries of the image indices and get the surrounding blues
    valid_indices = (y - 2 >= 0)
    left_blue = np.where(valid_indices, padded_image[x, y - 2], 0)

    valid_indices = (x - 2 >= 0)
    top_blue = np.where(valid_indices, padded_image[x - 2, y], 0)
    
    valid_indices = (x + 2 < image.shape[0])
    bottom_blue = np.where(valid_indices, padded_image[x + 2, y], 0)

    valid_indices = (y + 2 < image.shape[1])
    right_blue = np.where(valid_indices, padded_image[x, y + 2], 0)

    # Add the surrounding blue pixels together
    numerator = left_blue + right_blue + top_blue + bottom_blue

    # Get the divisor in case if one of them has a 0 for proper averages
    divisor = ((left_blue != 0).astype(int) + 
            (right_blue != 0).astype(int) + 
            (top_blue != 0).astype(int) + 
            (bottom_blue != 0).astype(int))

    # Get the average and place it in the original image
    image.put(np.ravel_multi_index(red_index.T, image.shape), numerator // divisor)
    
    # Performing masking if wanted
    if __mask__:
        mask = mask_by_color('red', pattern_dict, image)
        result = np.where(mask, image, 0)
        return result

def transform_IR_to_red(IR_pos_index, IR_neg_index, image, pattern_dict, __mask__=False):
    pos_x = IR_pos_index[:, 0]
    pos_y = IR_pos_index[:, 1]
    neg_x = IR_neg_index[:, 0]
    neg_y = IR_neg_index[:, 1]

    # Pad original image with 2 rows and columns of zero on the right and bottom side
    padded_image = np.pad(image, (0, 1), mode='constant')

    # Get valid indices within the boundaries of the image indices and get the
    # diagonal reds whether it be positive or negative following the index arrays
    valid_indices = (pos_x + 1 < image.shape[0]) & (pos_y - 1 >= 0)
    top_right_val = np.where(valid_indices, padded_image[pos_x + 1, pos_y - 1], 0)

    valid_indices = (pos_x - 1 >= 0) & (pos_y + 1 < image.shape[1])
    bottom_left_val = np.where(valid_indices, padded_image[pos_x - 1, pos_y + 1], 0)

    valid_indices = (neg_x - 1 >= 0) & (neg_y - 1 >= 0)
    top_left_val = np.where(valid_indices, padded_image[neg_x - 1, neg_y - 1], 0)

    valid_indices = (neg_x + 1 < image.shape[0]) & (neg_y + 1 < image.shape[1])
    bottom_right_val = np.where(valid_indices, padded_image[neg_x + 1, neg_y + 1], 0)

    # Get average for negative diagonal
    numerator = top_left_val + bottom_right_val
    divisor = ((top_left_val != 0).astype(np.uint8) + (bottom_right_val != 0).astype(np.uint8))
    answer = numerator // divisor

    # Put into original image
    image.put(np.ravel_multi_index(IR_neg_index.T, image.shape), answer)
    
    # Get average for positive diagonal
    numerator = top_right_val + bottom_left_val
    divisor = ((top_right_val != 0).astype(np.uint8) + (bottom_left_val != 0).astype(np.uint8))
    
    # Handle edge case for IR at bottom right with no positive diagonal red
    divisor[divisor == 0] = 1

    answer = numerator // divisor
    answer[answer == 0] = image[-1, -1]

    # Put back into original image
    image.put(np.ravel_multi_index(IR_pos_index.T, image.shape), answer)

    if __mask__:
        mask = mask_by_color('IR', pattern_dict, image)
        result = np.where(mask, image, 0)
        return result

def upsample(color_channel, kernel_size):
    """
    Upsample a numpy matrix
    Used to fill IR pixels.
    """
    color_channel = np.kron(color_channel, np.ones(kernel_size)).astype(np.uint16)
    return color_channel

def fill_zeroes(color_channel, kernel_size, color_position):
    """
    Fill 0 values of a 2D numpy array with
    color pixels in the bottom right of a certain kernel size.
    Used to fill IR pixels.
    IR pixels location is at the bottom right in a 2x2 kernel.
    """
    bottom_right_val = color_channel[color_position[0]::kernel_size[0], color_position[1]::kernel_size[1]]
    color_channel = np.kron(bottom_right_val, np.ones((2, 2))).astype(np.uint16)
    return color_channel

def subtract_IR(convolved_image, IR_color_channel, red_coeff=0.5, green_coeff=0.2, blue_coeff=0.3):
    red, green_red, green_blue, blue = get_bayer_indices('bggr')
    
    red_indices = np.column_stack(np.where(mask_by_color(red, convolved_image, (2, 2))))
    green_indices = np.column_stack(np.where(mask_by_color((green_red, green_blue), convolved_image, (2, 2))))
    blue_indices = np.column_stack(np.where(mask_by_color(blue, convolved_image, (2, 2))))
    # red_indices = get_color_indices('red', pattern, convolved_image, (2, 2))
    # green_indices = get_color_indices('green', pattern, convolved_image, (2, 2))
    # blue_indices = get_color_indices('blue', pattern, convolved_image, (2, 2))
    
    convolved_image[red_indices[:, 0], red_indices[:, 1]] -= (red_coeff * IR_color_channel[red_indices[:, 0], red_indices[:, 1]]).astype(np.uint16)
    convolved_image[green_indices[:, 0], green_indices[:, 1]] -= (green_coeff * IR_color_channel[green_indices[:, 0], green_indices[:, 1]]).astype(np.uint16)
    convolved_image[blue_indices[:, 0], blue_indices[:, 1]] -= (blue_coeff * IR_color_channel[blue_indices[:, 0], blue_indices[:, 1]]).astype(np.uint16)


# Perform Guided upsampling by Gaussian Filtering
def guided_upsampling(image, guide, sigma=1.0):
    # Compute guidance image from the low-resolution image

    # Compute the Gaussian filter based on the guide image
    guide_filtered = cv2.GaussianBlur(guide, (0, 0), sigma)

    # Upsample the target image using inter_area interpolation
    upscaled_image = cv2.resize(image, guide_filtered.shape[::-1], interpolation=cv2.INTER_AREA)

    # Adjust each pixel in the upscaled image based on the corresponding pixel in the filtered guide image
    adjusted_image = upscaled_image + (guide_filtered - upscaled_image)

    # Clip the pixel values to the valid range [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 255)

    # Convert back to uint8
    adjusted_image = adjusted_image.astype(np.uint8)

    # save_to_txt('guided_upsample.txt', adjusted_image)
    return adjusted_image


if __name__ == '__main__':
    raw_path = 'raw/Bright_color.raw'
    bayer = np.fromfile(raw_path, dtype='uint16', sep='')
    bayer = bayer.reshape((1944, 2592))
    sub_arrays = {}
    # Arranged in the order of (x-coord), (y-coords)
    rgb_ir_indices = {
                    'blue'  : ((0, 2), (0, 2)),
                    'red'   : ((0, 2), (2, 0)),
                    'green' : ((0, 0, 1, 1, 2, 2, 3, 3), (1, 3, 0, 2, 1, 3, 0, 2)),
                    'IR'    : ((1, 1, 3, 3), (1, 3, 1, 3)),
                    'IR_pos': ((1, 3), (1, 3)),
                    'IR_neg': ((1, 3), (3, 1))
                 }
    # save_to_txt('bayer_image.txt', bayer)

    IR_color_channel = extract_IR(bayer, rgb_ir_indices)

    # save_to_txt('adjusted_IR_image_channel.txt', IR_color_channel)

    filled_IR_color_channel = fill_zeroes(IR_color_channel, (2, 2), (1, 1))

    # save_to_txt('adjusted_filled_IR_image.txt', filled_IR_color_channel)

    # # IR_indices = get_color_indices('IR', rgb_ir_indices, bayer)
    IR_pos_indices = get_color_indices('IR_pos', rgb_ir_indices, bayer)
    IR_neg_indices = get_color_indices('IR_neg', rgb_ir_indices, bayer)
    transform_IR_to_red(IR_pos_indices, IR_neg_indices, bayer, rgb_ir_indices)

    red_indices = get_color_indices('red', rgb_ir_indices, bayer)
    transform_red_to_blue(red_indices, bayer, rgb_ir_indices)


    # save_to_txt('altered_averaged_IR_values.txt', result)

    # # bayer[red_indices[:, 0], red_indices[:, 1]] = changed_red_to_blue[red_indices[:, 0], red_indices[:, 1]]
    # # bayer[IR_indices[:, 0], IR_indices[:, 1]] = changed_IR_to_red[IR_indices[:, 0], IR_indices[:, 1]]

    start_time = time.time()
    # # save_to_txt('Modified_bayer.txt', bayer)
    subtract_IR(bayer, filled_IR_color_channel)
    # # save_to_txt('inplace_bayer_interpolation.txt', bayer)
    save_to_txt('adjusted_subtracted_IR_from_bayer.txt', bayer)

    end_time = time.time() - start_time
    print(f'Time elapsed {end_time:.3f}')


    