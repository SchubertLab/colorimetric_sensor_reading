import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Functions ------------------------------------------------------------------------------------------------
def create_circular_mask(h, w, center=None, radius=None):
    """ Returns an image with a circular mask on the specified coordinates.

            If center is none uses the center of the image,
            if radius is none se the smallest distance between the center and image walls
            Parameters
            ----------
            h : int
                image height in pixels
            w : int
                image width in pixels
            center : [int, int], optional
                The coordinates of the circle center in pixels (default is None)
            radius : int, optional
                Circle radius in pixels (default is None)
    """
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def polynomial_kernel(dim_x, dim_y, n_power_x=1, n_power_y=1, scale_x=1, scale_y=1, offset_x=0, offset_y=0):
    """ Returns an image with a polynomial kernel noise distribution.

        Parameters
        ----------
        dim_x : int
            image height in pixels
        dim_y : int
            image width in pixels
        n_power_x, n_power_y : int, int
            power for the polynomial functions in x and y
        scale_x, scale_y : int, int
        offset_x, offset_y: offset of the polynomial function
    """
    # polynomial for x dimension
    kernel_x = np.zeros((dim_y, dim_x))
    x = np.arange(dim_x) / dim_x
    x_polynomial = ((scale_x * x) ** n_power_x) + offset_x
    for i in range(dim_y):
        kernel_x[i:i + 1, :] = x_polynomial

    # polynomial for y dimension
    kernel_y = np.zeros((dim_x, dim_y))
    y = np.arange(dim_y) / dim_y
    y_polynomial = ((scale_y * y) ** n_power_y) + offset_y
    for i in range(dim_x):
        kernel_y[i:i + 1, :] = y_polynomial
    kernel_y = np.transpose(kernel_y)

    # Average x & y kernels
    kernel_xy = np.mean(np.array([kernel_x, kernel_y]), axis=0)

    # Min-Max Normalization [0-1]
    kernel_xy = (kernel_xy - np.min(kernel_xy)) / (np.max(kernel_xy) - np.min(kernel_xy))

    return kernel_xy


def radial_kernel(dim_x, dim_y, center_x, center_y, scale_radius_x, scale_radius_y):
    """ Returns an image with an elliptical kernel noise distribution.

        Parameters
        ----------
        dim_x : int
            image height in pixels
        dim_y : int
            image width in pixels
        center_x, center_y : int, int
            offset of the center of the elliptical distribution
        scale_radius_x, scale_radius_y : int, int
    """
    dist_to_center_x_array = [(((x - center_x) / dim_x) ** 2 * scale_radius_x) for x in np.arange(dim_x)]
    dist_to_center_y_array = [(((y - center_y) / dim_y) ** 2 * scale_radius_y) for y in np.arange(dim_y)]

    kernel_x = np.zeros((dim_y, dim_x))
    for i in range(dim_y):
        kernel_x[i:i + 1, :] = dist_to_center_x_array

    kernel_y = np.zeros((dim_x, dim_y))
    for i in range(dim_x):
        kernel_y[i:i + 1, :] = dist_to_center_y_array
    kernel_y = np.transpose(kernel_y)

    distance_kernel = kernel_x + kernel_y + np.ones(kernel_x.shape)
    light_kernel = np.divide(np.ones(distance_kernel.shape), distance_kernel)

    # Min-Max Normalization [0-1]
    light_kernel = (light_kernel - np.min(light_kernel)) / (np.max(light_kernel) - np.min(light_kernel))

    return light_kernel


def spatial_gain(image_in, alpha, beta, gamma, kernel):
    """ Returns the input image merged with a noise kernel.

        Parameters
        ----------
        image_in : image
            input image to be modified
        alpha, beta, gamma : int, int, int
            intensity factors to merge images
        kernel : image
            noise distribution
    """
    img_out = np.zeros(image_in.shape, dtype=int)

    alpha_m = kernel * alpha
    beta_m = kernel * beta
    gamma_m = kernel * gamma

    img_out[:, :, 0] = image_in[:, :, 0] + alpha_m
    img_out[:, :, 1] = image_in[:, :, 1] + beta_m
    img_out[:, :, 2] = image_in[:, :, 2] + gamma_m

    img_out = np.clip(img_out, 0, 255)
    return img_out


def rotate_flip(input_img):
    """ Returns the input image randomly rotated or flipped.
    """
    rotate_or_flip = random.randint(0, 2)
    if rotate_or_flip == 0:
        flip_ud_lr = random.randint(0, 1)
        if flip_ud_lr == 0:
            input_img = np.flipud(input_img)
        else:
            input_img = np.fliplr(input_img)
    elif rotate_or_flip == 1:
        rot_c = random.randint(0, 2)
        if rot_c == 0:
            input_img = np.rot90(input_img)
        if rot_c == 1:
            input_img = np.rot90(input_img)
            input_img = np.rot90(input_img)
        else:
            input_img = np.rot90(input_img)
            input_img = np.rot90(input_img)
            input_img = np.rot90(input_img)
    return input_img


# ---------------------------------------------------------------------------------------------------------------------

def run_augmentations_pipeline(path_dataset, df_gt_data, output_circle_ref, N_samples_per_type,
                               interp_kernel, interp_image, save_path_gt, save_path_noise, save_path_kernel):
    """ Main function to generate image augmentations based on random polynomial, elliptical and combined kernels.
        Saves images in the following directories: ground truth, noisy images, light kernel
        Returns a df with the dataset characteristics

    """
    dict_dataset = {
        'ph_value': [],
        'temperature_value': [],
        'experiment_id': [],
        'sensor_name': [],
        'raw_image_name': [],
        'frame': [],
        'path_noise': [],
        'path_gt': [],
        'path_light_kernel': [],
        'type_of_kernel': [],
        'alpha_R_channel': [],
        'beta_G_channel': [],
        'gamma_B_channel': [],
        'altered_img_name': [],
    }

    spot_vals = np.array([0, 1, 2, 3, 4, 5, 6])
    for s in spot_vals:
        dict_dataset['spot_' + str(s)] = []

    counter = 0
    for index, row in df_gt_data.iterrows():
        temp_img = row['image_name'] + '.png'
        print(temp_img, counter)
        # Load temp image
        sensor_bgr = cv2.imread(path_dataset + temp_img)
        sensor_rgb = cv2.cvtColor(sensor_bgr, cv2.COLOR_BGR2RGB)

        # Outside Mask
        outer_mask_ref = create_circular_mask(
            h=sensor_rgb.shape[0],
            w=sensor_rgb.shape[1],
            center=output_circle_ref[:2],
            radius=output_circle_ref[2],
        )

        for i in range(N_samples_per_type):
            # Light Kernel intensity
            alpha_beta_gamma = np.arange(0, 0.5, 0.01) * 255

            # Polynomial Samples  ----------------------------------------------------------------------------------
            polynom_array = np.array([1, 2, 3])
            linear_scale = np.arange(-10, 10, 1)
            linear_offset = np.arange(-20, 20, 1)

            polynomial_light_kernel = polynomial_kernel(
                dim_x=sensor_rgb.shape[0],
                dim_y=sensor_rgb.shape[1],
                n_power_x=np.random.choice(polynom_array, 1)[0],
                n_power_y=np.random.choice(polynom_array, 1)[0],
                scale_x=np.random.choice(linear_scale, 1)[0],
                scale_y=np.random.choice(linear_scale, 1)[0],
                offset_x=np.random.choice(linear_offset, 1)[0],
                offset_y=np.random.choice(linear_offset, 1)[0],
            )
            polynomial_light_kernel = rotate_flip(polynomial_light_kernel)

            temp_sign = np.random.choice([1, -1], 1)[0]
            poly_alpha = int(np.random.choice(alpha_beta_gamma, 1)[0]) * temp_sign
            poly_beta = int(np.random.choice(alpha_beta_gamma, 1)[0]) * temp_sign
            poly_gamma = int(np.random.choice(alpha_beta_gamma, 1)[0]) * temp_sign

            sensor_polynomial = spatial_gain(
                image_in=sensor_rgb,
                alpha=poly_alpha,
                beta=poly_beta,
                gamma=poly_gamma,
                kernel=polynomial_light_kernel,
            )
            sensor_polynomial[~outer_mask_ref] = 0

            # Radial Samples ----------------------------------------------------------------------------------
            center_xy = np.arange(-int(sensor_rgb.shape[0] / 2), sensor_rgb.shape[0], 100)
            intensity_arr = np.arange(-25, 100, 5)
            scale_radius = np.arange(1, 40, 0.5)

            radial_light_kernel = radial_kernel(
                dim_x=sensor_rgb.shape[0],
                dim_y=sensor_rgb.shape[1],
                center_x=np.random.choice(center_xy, 1)[0],
                center_y=np.random.choice(center_xy, 1)[0],
                scale_radius_x=np.random.choice(scale_radius, 1)[0],
                scale_radius_y=np.random.choice(scale_radius, 1)[0],
            )
            radial_light_kernel = rotate_flip(radial_light_kernel)

            temp_sign_rad = np.random.choice([1, -1], 1)[0]
            rad_alpha = int(np.random.choice(alpha_beta_gamma, 1)[0]) * temp_sign_rad
            rad_beta = int(np.random.choice(alpha_beta_gamma, 1)[0]) * temp_sign_rad
            rad_gamma = int(np.random.choice(alpha_beta_gamma, 1)[0]) * temp_sign_rad

            sensor_radial = spatial_gain(
                image_in=sensor_rgb,
                alpha=rad_alpha,
                beta=rad_beta,
                gamma=rad_gamma,
                kernel=radial_light_kernel,
            )
            sensor_radial[~outer_mask_ref] = 0

            # Polynomial + Radial Kernel Samples -----------------------------------------------------------
            polynomial_light_kernel_2 = polynomial_kernel(
                dim_x=sensor_rgb.shape[0],
                dim_y=sensor_rgb.shape[1],
                n_power_x=np.random.choice(polynom_array, 1)[0],
                n_power_y=np.random.choice(polynom_array, 1)[0],
                scale_x=np.random.choice(linear_scale, 1)[0],
                scale_y=np.random.choice(linear_scale, 1)[0],
                offset_x=np.random.choice(linear_offset, 1)[0],
                offset_y=np.random.choice(linear_offset, 1)[0],
            )
            polynomial_light_kernel_2 = rotate_flip(polynomial_light_kernel_2) * 0.5

            radial_light_kernel_2 = radial_kernel(
                dim_x=sensor_rgb.shape[0],
                dim_y=sensor_rgb.shape[1],
                center_x=np.random.choice(center_xy, 1)[0],
                center_y=np.random.choice(center_xy, 1)[0],
                scale_radius_x=np.random.choice(scale_radius, 1)[0],
                scale_radius_y=np.random.choice(scale_radius, 1)[0],
            )
            radial_light_kernel_2 = rotate_flip(radial_light_kernel_2) * 0.5
            poly_radial_kernel = polynomial_light_kernel_2 + radial_light_kernel_2

            temp_sign = np.random.choice([1, -1], 1)[0]
            poly_rad_alpha = int(np.random.choice(alpha_beta_gamma, 1)[0]) * temp_sign
            poly_rad_beta = int(np.random.choice(alpha_beta_gamma, 1)[0]) * temp_sign
            poly_rad_gamma = int(np.random.choice(alpha_beta_gamma, 1)[0]) * temp_sign

            sensor_poly_radial = spatial_gain(
                image_in=sensor_rgb,
                alpha=poly_rad_alpha,
                beta=poly_rad_beta,
                gamma=poly_rad_gamma,
                kernel=poly_radial_kernel,
            )
            sensor_poly_radial[~outer_mask_ref] = 0

            # Save Images --------------------------------------------------------------------------------------------
            # Resize
            gt_small = cv2.resize(sensor_rgb, (250, 250), interpolation=interp_image)
            poly_crop_small = cv2.resize(sensor_polynomial.astype(np.uint8), (250, 250), interpolation=interp_image)
            radial_crop_small = cv2.resize(sensor_radial.astype(np.uint8), (250, 250), interpolation=interp_image)
            poly_rad_crop_small = cv2.resize(sensor_poly_radial.astype(np.uint8), (250, 250),
                                             interpolation=interp_image)

            polynomial_light_kernel = polynomial_light_kernel * 255
            polynomial_light_kernel[~outer_mask_ref] = 0
            poly_light_kernel_small = cv2.resize(polynomial_light_kernel.astype(np.uint8), (250, 250),
                                                 interpolation=interp_kernel)
            radial_light_kernel = radial_light_kernel * 255
            radial_light_kernel[~outer_mask_ref] = 0
            rad_light_kernel_small = cv2.resize(radial_light_kernel.astype(np.uint8), (250, 250),
                                                interpolation=interp_kernel)

            poly_rad_light_kernel = poly_radial_kernel * 255
            poly_rad_light_kernel[~outer_mask_ref] = 0
            poly_rad_light_kernel_small = cv2.resize(poly_rad_light_kernel.astype(np.uint8), (250, 250),
                                                     interpolation=interp_kernel)

            # Save polynomial -----------------------------------------------------------------------------------------
            new_image_name = 'poly' + str(counter) + '_' + temp_img
            plt.imsave(save_path_noise + new_image_name, poly_crop_small)
            plt.imsave(save_path_gt + new_image_name, gt_small)
            plt.imsave(save_path_kernel + new_image_name, poly_light_kernel_small)

            dict_dataset['ph_value'].append(row['ph_value'])
            dict_dataset['temperature_value'].append(row['temperature_value'])
            dict_dataset['experiment_id'].append(row['experiment_id'])
            dict_dataset['frame'].append(row['frame'])
            dict_dataset['sensor_name'].append(row['sensor_name'])
            dict_dataset['raw_image_name'].append(row['image_name'])
            for s in spot_vals:
                dict_dataset['spot_' + str(s)].append(row['spot_' + str(s)])
            # --
            dict_dataset['path_noise'].append(save_path_noise + new_image_name)
            dict_dataset['path_gt'].append(save_path_gt + new_image_name)
            dict_dataset['path_light_kernel'].append(save_path_kernel + new_image_name)
            dict_dataset['type_of_kernel'].append('polynomial')
            dict_dataset['alpha_R_channel'].append(poly_alpha)
            dict_dataset['beta_G_channel'].append(poly_beta)
            dict_dataset['gamma_B_channel'].append(poly_gamma)
            dict_dataset['altered_img_name'].append(new_image_name)

            # Save radial -----------------------------------------------------------------------------------------
            new_image_name = 'rad' + str(counter) + '_' + temp_img
            plt.imsave(save_path_noise + 'rad' + str(counter) + '_' + temp_img, radial_crop_small)
            plt.imsave(save_path_gt + 'rad' + str(counter) + '_' + temp_img, gt_small)
            plt.imsave(save_path_kernel + 'rad' + str(counter) + '_' + temp_img, rad_light_kernel_small)

            dict_dataset['ph_value'].append(row['ph_value'])
            dict_dataset['temperature_value'].append(row['temperature_value'])
            dict_dataset['experiment_id'].append(row['experiment_id'])
            dict_dataset['frame'].append(row['frame'])
            dict_dataset['sensor_name'].append(row['sensor_name'])
            dict_dataset['raw_image_name'].append(temp_img)
            for s in spot_vals:
                dict_dataset['spot_' + str(s)].append(row['spot_' + str(s)])
            # --
            dict_dataset['path_noise'].append(save_path_noise + new_image_name)
            dict_dataset['path_gt'].append(save_path_gt + new_image_name)
            dict_dataset['path_light_kernel'].append(SAVE_LIGHT_KERNEL_FULL + new_image_name)
            dict_dataset['type_of_kernel'].append('radial')
            dict_dataset['alpha_R_channel'].append(rad_alpha)
            dict_dataset['beta_G_channel'].append(rad_beta)
            dict_dataset['gamma_B_channel'].append(rad_gamma)
            dict_dataset['altered_img_name'].append(new_image_name)

            # Save poly-rad  -----------------------------------------------------------------------------------------
            new_image_name = 'polyrad' + str(counter) + '_' + temp_img
            plt.imsave(save_path_noise + 'polyrad' + str(counter) + '_' + temp_img, poly_rad_crop_small)
            plt.imsave(save_path_gt + 'polyrad' + str(counter) + '_' + temp_img, gt_small)
            plt.imsave(save_path_kernel + 'polyrad' + str(counter) + '_' + temp_img, poly_rad_light_kernel_small)

            dict_dataset['ph_value'].append(row['ph_value'])
            dict_dataset['temperature_value'].append(row['temperature_value'])
            dict_dataset['experiment_id'].append(row['experiment_id'])
            dict_dataset['frame'].append(row['frame'])
            dict_dataset['sensor_name'].append(row['sensor_name'])
            dict_dataset['raw_image_name'].append(temp_img)
            for s in spot_vals:
                dict_dataset['spot_' + str(s)].append(row['spot_' + str(s)])
            # --
            dict_dataset['path_noise'].append(save_path_noise + new_image_name)
            dict_dataset['path_gt'].append(save_path_gt + new_image_name)
            dict_dataset['path_light_kernel'].append(save_path_kernel + new_image_name)
            dict_dataset['type_of_kernel'].append('polyrad')
            dict_dataset['alpha_R_channel'].append(poly_rad_alpha)
            dict_dataset['beta_G_channel'].append(poly_rad_beta)
            dict_dataset['gamma_B_channel'].append(poly_rad_gamma)
            dict_dataset['altered_img_name'].append(new_image_name)
            counter += 1

    df_dataset = pd.DataFrame.from_dict(dict_dataset)
    return df_dataset


if __name__ == "__main__":
    # CONFIGURATIONS
    PATH_DF_DATASET = '../../data/processed/ground_truths/df_generated_data.csv'

    SENSOR_COORDS = '../../data/raw/template/sensing_area_coords.csv'

    PATH_DATASET = '../../data/processed/ground_truths/generated_data/'
    SAVE_DF_PATH = '../../data/processed/color_alterations/color_alterations/'
    NAME_DATAFRAME = 'df_noisy_data.csv'

    SAVE_ALTERED_FULL = '../../data/processed/color_alterations/noisy_images/'
    SAVE_GT_FULL = '../../data/processed/color_alterations/ground_truths/'
    SAVE_LIGHT_KERNEL_FULL = '../../data/processed/color_alterations/light_kernel/'

    # Template Data

    # Sensor Template Data
    refs_dict = {
        'R1': [[505, 602], [269, 413]],
        'R2': [[505, 602], [628, 772]],
    }
    output_circle_ref = [520, 476, 383.0]

    # Augmented Samples Radial and Polynomial
    N_samples_per_type = 1

    INTERPOLATION_KERNEL = cv2.INTER_NEAREST
    INTERPOLATION_IMAGE = cv2.INTER_AREA

    df_gt_data = pd.read_csv(PATH_DF_DATASET)

    # ----------------------------------------------------------------------------------------------------
    # AUGMENT DATASET
    df_dataset = run_augmentations_pipeline(
        path_dataset=PATH_DATASET,
        df_gt_data=df_gt_data,
        output_circle_ref=output_circle_ref,
        N_samples_per_type=N_samples_per_type,
        interp_kernel=INTERPOLATION_KERNEL,
        interp_image=INTERPOLATION_IMAGE,
        save_path_gt=SAVE_GT_FULL,
        save_path_noise=SAVE_ALTERED_FULL,
        save_path_kernel=SAVE_LIGHT_KERNEL_FULL,
    )
    df_dataset.to_csv(SAVE_DF_PATH + NAME_DATAFRAME, index=False)
