# File: helpers.py
# Description: Numpy helpers for image processing
# Created: 2021/10/22 20:34
# Author: Qiu Jueqin (qiujueqin@gmail.com)

import numpy as np
import os
import math
from scipy.ndimage import zoom, gaussian_filter
import cv2


def get_bayer_indices(pattern):
    """
    Get (x_start_idx, y_start_idx) for R, Gr, Gb, and B channels
    in Bayer array, respectively
    """
    return {'gbrg': ((0, 1), (1, 1), (0, 0), (1, 0)),
            'rggb': ((0, 0), (1, 0), (0, 1), (1, 1)),
            'bggr': ((1, 1), (0, 1), (1, 0), (0, 0)),
            'grbg': ((1, 0), (0, 0), (1, 1), (0, 1))}[pattern.lower()]


def split_bayer(bayer_array, bayer_pattern):
    """
    Split R, Gr, Gb, and B channels sub-array from a Bayer array
    :param bayer_array: np.ndarray(H, W)
    :param bayer_pattern: 'gbrg' | 'rggb' | 'bggr' | 'grbg'
    :return: 4-element list of R, Gr, Gb, and B channel sub-arrays, each is an np.ndarray(H/2, W/2)
    """
    rggb_indices = get_bayer_indices(bayer_pattern)

    sub_arrays = []
    for idx in rggb_indices:
        x0, y0 = idx
        sub_arrays.append(
            bayer_array[y0::2, x0::2]
        )

    return sub_arrays


def reconstruct_bayer(sub_arrays, bayer_pattern):
    """
    Inverse implementation of split_bayer: reconstruct a Bayer array from a list of
        R, Gr, Gb, and B channel sub-arrays
    :param sub_arrays: 4-element list of R, Gr, Gb, and B channel sub-arrays, each np.ndarray(H/2, W/2)
    :param bayer_pattern: 'gbrg' | 'rggb' | 'bggr' | 'grbg'
    :return: np.ndarray(H, W)
    """
    rggb_indices = get_bayer_indices(bayer_pattern)

    height, width = sub_arrays[0].shape
    bayer_array = np.empty(shape=(2 * height, 2 * width), dtype=sub_arrays[0].dtype)

    for idx, sub_array in zip(rggb_indices, sub_arrays):
        x0, y0 = idx
        bayer_array[y0::2, x0::2] = sub_array

    return bayer_array


def pad(array, pads, mode='reflect'):
    """
    Pad an array with given margins
    :param array: np.ndarray(H, W, ...)
    :param pads: {int, sequence}
        if int, pad top, bottom, left, and right directions with the same margin
        if 2-element sequence: (y-direction pad, x-direction pad)
        if 4-element sequence: (top pad, bottom pad, left pad, right pad)
    :param mode: padding mode, see np.pad
    :return: padded array: np.ndarray(H', W', ...)
    """
    if isinstance(pads, (list, tuple, np.ndarray)):
        if len(pads) == 2:
            pads = ((pads[0], pads[0]), (pads[1], pads[1])) + ((0, 0),) * (array.ndim - 2)
        elif len(pads) == 4:
            pads = ((pads[0], pads[1]), (pads[2], pads[3])) + ((0, 0),) * (array.ndim - 2)
        else:
            raise NotImplementedError

    return np.pad(array, pads, mode)


def crop(array, crops):
    """
    Crop an array by given margins
    :param array: np.ndarray(H, W, ...)
    :param crops: {int, sequence}
        if int, crops top, bottom, left, and right directions with the same margin
        if 2-element sequence: (y-direction crop, x-direction crop)
        if 4-element sequence: (top crop, bottom crop, left crop, right crop)
    :return: cropped array: np.ndarray(H', W', ...)
    """
    if isinstance(crops, (list, tuple, np.ndarray)):
        if len(crops) == 2:
            top_crop = bottom_crop = crops[0]
            left_crop = right_crop = crops[1]
        elif len(crops) == 4:
            top_crop, bottom_crop, left_crop, right_crop = crops
        else:
            raise NotImplementedError
    else:
        top_crop = bottom_crop = left_crop = right_crop = crops

    height, width = array.shape[:2]
    return array[top_crop: height - bottom_crop, left_crop: width - right_crop, ...]


def shift_array(padded_array, window_size):
    """
    Shift an array within a window and generate window_size**2 shifted arrays
    :param padded_array: np.ndarray(H+2r, W+2r)
    :param window_size: 2r+1
    :return: a generator of length (2r+1)*(2r+1), each is an np.ndarray(H, W), and the original
        array before padding locates in the middle of the generator
    """
    wy, wx = window_size if isinstance(window_size, (list, tuple)) else (window_size, window_size)
    assert wy % 2 == 1 and wx % 2 == 1, 'only odd window size is valid'

    height = padded_array.shape[0] - wy + 1
    width = padded_array.shape[1] - wx + 1

    for y0 in range(wy):
        for x0 in range(wx):
            yield padded_array[y0:y0 + height, x0:x0 + width, ...]


def gen_gaussian_kernel(kernel_size, sigma):
    if isinstance(kernel_size, (list, tuple)):
        assert len(kernel_size) == 2
        wy, wx = kernel_size
    else:
        wy = wx = kernel_size

    x = np.arange(wx) - wx // 2
    if wx % 2 == 0:
        x += 0.5

    y = np.arange(wy) - wy // 2
    if wy % 2 == 0:
        y += 0.5

    y, x = np.meshgrid(y, x)

    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def generic_filter(array, kernel):
    """
    Filter input image array with given kernel
    :param array: array to be filter: np.ndarray(H, W, ...), must be np.int dtype
    :param kernel: np.ndarray(h, w)
    :return: filtered array: np.ndarray(H, W, ...)
    """
    kh, kw = kernel.shape[:2]
    kernel = kernel.flatten()

    padded_array = pad(array, pads=(kh // 2, kw // 2))
    shifted_arrays = shift_array(padded_array, window_size=(kh, kw))

    filtered_array = np.zeros_like(array)
    weights = np.zeros_like(array)

    for i, shifted_array in enumerate(shifted_arrays):
        filtered_array += kernel[i] * shifted_array
        weights += kernel[i]

    filtered_array = (filtered_array / weights).astype(array.dtype)
    return filtered_array


def mean_filter(array, filter_size=3):
    """
    A faster reimplementation of the mean filter
    :param array: array to be filter: np.ndarray(H, W, ...)
    :param filter_size: int, diameter of the mean-filter
    :return: filtered array: np.ndarray(H, W, ...)
    """

    assert filter_size % 2 == 1, 'only odd filter size is valid'

    padded_array = pad(array, pads=filter_size // 2)
    shifted_arrays = shift_array(padded_array, window_size=filter_size)
    return (sum(shifted_arrays) / filter_size ** 2).astype(array.dtype)


def bilateral_filter(array, spatial_weights, intensity_weights_lut, right_shift=0):
    """
    A faster reimplementation of the bilateral filter
    :param array: array to be filter: np.ndarray(H, W, ...), must be np.int dtype
    :param spatial_weights: np.ndarray(h, w): predefined spatial gaussian kernel, where h and w are
        kernel height and width respectively
    :param intensity_weights_lut: a predefined exponential LUT that maps intensity distance to the weight
    :param right_shift: shift the multiplication result of the spatial- and intensity weights to the
        right to avoid integer overflow when multiply this result to the input array
    :return: filtered array: np.ndarray(H, W, ...)
    """
    filter_height, filter_width = spatial_weights.shape[:2]
    spatial_weights = spatial_weights.flatten()

    padded_array = pad(array, pads=(filter_height // 2, filter_width // 2))
    shifted_arrays = shift_array(padded_array, window_size=(filter_height, filter_width))

    bf_array = np.zeros_like(array)
    weights = np.zeros_like(array)

    for i, shifted_array in enumerate(shifted_arrays):
        intensity_diff = (shifted_array - array) ** 2
        weight = intensity_weights_lut[intensity_diff] * spatial_weights[i]
        weight = np.right_shift(weight, right_shift)  # to avoid overflow

        bf_array += weight * shifted_array
        weights += weight

    bf_array = (bf_array / weights).astype(array.dtype)

    return bf_array

# File: helpers.py
# Description: Helpers for RGB-IR to bggr bayer conversion
# Created: 2023/05/22 00:54
# Author: Samuel Theofie (samueltheofie@gmail.com)

def save_to_txt(name, obj):
    OUTPUT_DIR = './output'
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
    return np.where(mask)


def transform_red_to_blue(red_index, image, pattern_dict, __mask__=False):
    x = red_index[0]
    y = red_index[1]

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
    image.put(np.ravel_multi_index(red_index, image.shape), numerator // divisor)
    
    # Performing masking if wanted
    if __mask__:
        mask = mask_by_color('red', pattern_dict, image)
        result = np.where(mask, image, 0)
        return result


def transform_IR_to_red(IR_pos_index, IR_neg_index, image, pattern_dict, __mask__=False):
    pos_x = IR_pos_index[0]
    pos_y = IR_pos_index[1]
    neg_x = IR_neg_index[0]
    neg_y = IR_neg_index[1]

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
    image.put(np.ravel_multi_index(IR_neg_index, image.shape), answer)
    
    # Get average for positive diagonal
    numerator = top_right_val + bottom_left_val
    divisor = ((top_right_val != 0).astype(np.uint8) + (bottom_left_val != 0).astype(np.uint8))
    
    # Handle edge case for IR at bottom right with no positive diagonal red
    divisor[divisor == 0] = 1

    answer = numerator // divisor
    answer[answer == 0] = image[-1, -1]

    # Put back into original image
    image.put(np.ravel_multi_index(IR_pos_index, image.shape), answer)

    if __mask__:
        mask = mask_by_color('IR', pattern_dict, image)
        result = np.where(mask, image, 0)
        return result


def subtract_IR(convolved_image, IR_color_channel, red_coeff=0.717, green_coeff=0.22, blue_coeff=0.375):
    resized_IR = np.kron(IR_color_channel, np.ones((2, 2)))
    red, green_red, green_blue, blue = get_bayer_indices('bggr')
    red_subtract = red_coeff * resized_IR
    green_subtract = green_coeff * resized_IR
    blue_subtract = blue_coeff * resized_IR

    red_mask = mask_by_color(red, convolved_image, (2, 2))
    green_mask = mask_by_color((green_red, green_blue), convolved_image, (2, 2))
    blue_mask = mask_by_color(blue, convolved_image, (2, 2))
    
    red_subtract = np.where(red_mask, red_subtract.astype(np.uint32), 0)
    green_subtract = np.where(green_mask, green_subtract.astype(np.uint32), 0)
    blue_subtract = np.where(blue_mask, blue_subtract.astype(np.uint32), 0)
    
    convolved_image -= (red_subtract + green_subtract + blue_subtract)


# Perform Guided upsampling by Gaussian Filtering
def guided_upsampling(target_image, guide_image, zoom_lvl=2, sigma=1.0):
    # Upscale ir image
    upscaled_ir_image = zoom(target_image, zoom_lvl, order=0)

    # Apply Gaussian filtering to the guide image
    smoothed_guide = gaussian_filter(guide_image, sigma)

    # Compute the guidance map
    guidance_map = smoothed_guide - upscaled_ir_image 

    # Apply the guidance map to the upsampled image
    output_image = upscaled_ir_image + guidance_map

    return output_image