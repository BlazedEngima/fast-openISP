from collections import OrderedDict
import numpy as np
import os
import math
import time
import cv2

np.set_printoptions(
    suppress=True, 
    formatter={
        'int_kind':'{:d}'.format
    }
)

# Helper function to convert numpy array into a txt file
def save_to_txt(name, obj):
    output_path = os.path.join(OUTPUT_DIR, name)
    np.savetxt(output_path, obj, fmt='%d')

# Gets a mask at the coordinate positions of the color sensors
def mask_by_color(color):
    indices = rgb_ir_indices[color]
    small_mask = np.zeros((4, 4), dtype=bool)
    for idx in indices:
        small_mask[idx] = True

    repetitions = (
                    math.ceil(bayer.shape[0] / small_mask.shape[0]), 
                    math.ceil(bayer.shape[1] / small_mask.shape[1])
                   )

    mask = np.tile(small_mask, repetitions)

    return mask

# Splits the main raw file into the color sub-arrays
def get_color_sub_arrays():
    for color in rgb_ir_indices:
        # start_time = time.time()
        mask = mask_by_color(color)

        color_array = np.where(mask, bayer, 0)
        sub_arrays[color] = color_array
        # end_time = time.time() - start_time
        # print(f'{color}_array done. Time elapsed {end_time:.3f}s')

        # save_to_txt(f'{color}.txt', color_array)

# Linearly interpolates a sub-array on columns with a certain color argument 
def interpolate_vertical(image, color):
    coords = rgb_ir_indices[color]
    interpolation_axis = set(coords[i][0] for i in range(len(coords)))
    interpolation_axis = list(interpolation_axis)
    y = np.arange(image.shape[0])

    col_mask = np.zeros((image.shape[0], 4), dtype=bool)
    col_mask[:, interpolation_axis] = True

    repetitions = (1, math.ceil(image.shape[1] / col_mask.shape[1]))

    mask = np.tile(col_mask, repetitions)
    
    interpolated_matrix = np.zeros_like(image)
    for col in range(image.shape[1]):
        x = image[:, col]
        non_zero_indices = np.where(x != 0)[0]

        if len(non_zero_indices) > 0:
            interpolated_col = np.interp(y, non_zero_indices, x[non_zero_indices])
        else:
            interpolated_col = np.zeros_like(x)

        interpolated_matrix[:, col] = interpolated_col

    interpolated_image = np.where(mask, interpolated_matrix, 0)
    return interpolated_image

# Linearly interpolates a sub-array horizontally on rows with a certain color argument 
def interpolate_horizontal(image, color):
    coords = rgb_ir_indices[color]
    interpolation_axis = set(coords[i][0] for i in range(len(coords)))
    interpolation_axis = list(interpolation_axis)
    x = np.arange(image.shape[1])

    row_mask = np.zeros((4, image.shape[1]), dtype=bool)
    row_mask[interpolation_axis] = True

    repetitions = (math.ceil(image.shape[0] / row_mask.shape[0]), 1)

    mask = np.tile(row_mask, repetitions)
    
    interpolated_matrix = np.zeros_like(image)
    for row in range(image.shape[0]):
        y = image[row, :]
        non_zero_indices = np.where(y != 0)[0]

        if len(non_zero_indices) > 0:
            interpolated_row = np.interp(x, non_zero_indices, y[non_zero_indices])
        else:
            interpolated_row = np.zeros_like(y)

        interpolated_matrix[row, :] = interpolated_row

    interpolated_image = np.where(mask, interpolated_matrix, 0)
    return interpolated_image

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

# Gets the final interpolated row
def get_row_interpolation(color, image, upsample):
    mask = mask_by_color(color)
    masked_array = np.where(mask, upsample, 0)
    residue = image - masked_array
    residue = interpolate_horizontal(residue, color)

    result = residue + masked_array
    result = np.clip(result, 0, 255)

    # save_to_txt(f'green_{color}_row_interpolation.txt', result)
    return result

# Gets the final interpolated column
def get_column_interpolation(color, image, upsample):
    mask = mask_by_color(color)
    masked_array = np.where(mask, upsample, 0)
    residue = image - masked_array
    residue = interpolate_vertical(residue, color)

    result = residue + masked_array
    result = np.clip(result, 0, 255)

    # save_to_txt(f'green_{color}_column_interpolation.txt', result)
    return result


if __name__ == '__main__':
    OUTPUT_DIR = './output'
    raw_path = 'raw/Bright_color.raw'
    bayer = np.fromfile(raw_path, dtype='uint16', sep='')
    bayer = bayer.reshape((1944, 2592))
    sub_arrays = {}
    rgb_ir_indices = {
                    'blue'  : ((0, 0), (2, 2)),
                    'red'   : ((0, 2), (2, 0)),
                    'green' : ((0, 1), (0, 3), (1, 0), (1, 2),
                               (2, 1), (2, 3), (3, 0), (3, 2)),
                    'IR'    : ((1, 1), (1, 3), (3, 1), (3, 3))
                 }

    get_color_sub_arrays()
    
    for color in rgb_ir_indices:
        if color == 'green':
            continue

        interpolated_color_horizontal = interpolate_horizontal(sub_arrays['green'], color)
        interpolated_color_vertical = interpolate_vertical(sub_arrays['green'], color)

        guided_upsample_horizontal = guided_upsampling(sub_arrays[color], interpolated_color_horizontal)
        guided_upsample_vertical = guided_upsampling(sub_arrays[color], interpolated_color_vertical)

        interpolated_horizontal = get_row_interpolation(color, sub_arrays[color], guided_upsample_horizontal)
        interpolated_vertical = get_column_interpolation(color, sub_arrays[color], guided_upsample_vertical)

        # save_to_txt(f'interpolated_{color}_horizontal.txt', interpolated_horizontal)
        # save_to_txt(f'interpolated_{color}_vertical.txt', interpolated_vertical)
    