import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

today = date.today()


# Util Functions
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


def generate_random_noise(height, width, temp_coords, bool_threshold=0.65, radius_perc=1.28, random_scale_factor=1.1):
    """ Returns an image with noise and a boolean mask with pixes corresponding to the noise.

        Noise is added in the circular area specified in the temp_coords parameter

        Parameters
        ----------
        height : int
            image height in pixels
        width : int
            image width in pixels
        temp_coords : [int, int, int]
            coordinates of the circular mask (center x, center y, radius)
        bool_threshold : float, optional
            Noise threshold level of the image (lower threshold is higher noise) (default is 0.65)
        radius_perc : float, optional
            Increase the radius size (default is 1.28)
        random_scale_factor : float, optional
            Increase noise scale (default is 1.1)
    """

    temp_mask = create_circular_mask(
        h=height,
        w=width,
        center=temp_coords[:2],
        radius=temp_coords[2] * radius_perc,
    )

    random_matrix = temp_mask * np.random.rand(height, width) * random_scale_factor
    random_bool_mask = np.where(random_matrix > bool_threshold, True, False)
    random_matrix = random_matrix * 255

    noise = cv2.merge([random_matrix.astype(np.uint8), random_matrix.astype(np.uint8), random_matrix.astype(np.uint8)])

    return noise, random_bool_mask


def assign_circle_colors(input_image, circles_df, colors_dict, noise_perc=0.65, radius_perc=1.28):
    """ Returns two images of the sensor with new colors assigned to each area,
    one with noise and the same one without noise on the color sensing areas.

        Parameters
        ----------
        input_image : int
            image of the sensor to be modified
        circles_df : dataframe
            coordinates of the circular mask (center x, center y, radius) for each area
        colors_dict : dictionary
            colors assigned to each sensing area
        noise_perc : float, optional
            Noise threshold level of the image (lower threshold is higher noise) (default is 0.65)
        radius_perc : float, optional
            Enlarge circular mask (default is 1.28)
    """

    image_final = input_image.copy()
    image_final_no_noise = input_image.copy()
    for key, val in colors_dict.items():
        temp_area_coords = circles_df[key].to_numpy()
        temp_sensing_mask = create_circular_mask(
            h=input_image.shape[0],
            w=input_image.shape[1],
            center=temp_area_coords[:2],
            radius=temp_area_coords[2] * radius_perc,
        )
        r_mask = temp_sensing_mask * colors_dict[key][0]
        g_mask = temp_sensing_mask * colors_dict[key][1]
        b_mask = temp_sensing_mask * colors_dict[key][2]

        r_mask = r_mask.astype(np.uint8)
        g_mask = g_mask.astype(np.uint8)
        b_mask = b_mask.astype(np.uint8)
        image_sensing_spot = cv2.merge([r_mask, g_mask, b_mask])

        image_final_no_noise[np.where(temp_sensing_mask == True)] = image_sensing_spot[
            np.where(temp_sensing_mask == True)]

        noise_image, noise_mask = generate_random_noise(
            height=input_image.shape[0],
            width=input_image.shape[1],
            temp_coords=temp_area_coords,
            bool_threshold=noise_perc,
        )
        image_sensing_spot[np.where(noise_mask == True)] = noise_image[np.where(noise_mask == True)]
        image_final[np.where(temp_sensing_mask == True)] = image_sensing_spot[np.where(temp_sensing_mask == True)]

    return image_final, image_final_no_noise


# ----------------------------------------------------------------------------------------------------
# Main dataset generation function

def main_dataset_generation(experimental_df_path, path_sensor_image,
                            save_img_path_noise, save_img_path_without_noise,
                            circle_coords, colors_dict, dict_dataset,
                            experiment_name,
                            ):
    # Get Temperature Data
    df_temperature = pd.read_csv(experimental_df_path)

    df_temperature_norm = df_temperature.copy()

    spot_vals = np.array([0, 1, 2, 3, 4, 5, 6])
    channels = ['r_channel', 'g_channel', 'b_channel',
                'h_channel', 's_channel', 'v_channel',
                'grayscale']

    sensor_names = df_temperature['sensor_name'].unique()
    n_sensors = len(sensor_names)

    for s in spot_vals:
        dict_dataset['spot_' + str(s) + '_r'] = []
        dict_dataset['spot_' + str(s) + '_g'] = []
        dict_dataset['spot_' + str(s) + '_b'] = []

    # Min Max normalization and rescaling of color channels
    for sensor_i in sensor_names:
        for spot_i in spot_vals:
            current_sensor_df = df_temperature[
                (df_temperature['sensor_name'] == sensor_i) & (df_temperature['sensor_spot'] == spot_i)]

            ref_sensor = df_temperature[
                (df_temperature['sensor_name'] == REF_SENSOR) & (df_temperature['sensor_spot'] == spot_i)]

            for channel_i in channels:
                # channel min_max normalization
                scale = 1 / (current_sensor_df[channel_i].max() - current_sensor_df[channel_i].min())
                current_sensor_df[channel_i] = (current_sensor_df[channel_i] - current_sensor_df[
                    channel_i].min()) * scale
                # scale to ref sensor color
                ref_sensor_min = ref_sensor[channel_i].min()
                ref_sensor_max = ref_sensor[channel_i].max()
                current_sensor_df[channel_i] = (current_sensor_df[channel_i] * (
                        ref_sensor_max - ref_sensor_min)) + ref_sensor_min

            df_temperature_norm.loc[(df_temperature_norm['sensor_name'] == sensor_i)
                                    & (df_temperature_norm['sensor_spot'] == spot_i)] = current_sensor_df

    # ----------------------------------------------------------------------------------------------------
    # Load template image
    sensor_bgr = cv2.imread(path_sensor_image)
    sensor_rgb_big = cv2.cvtColor(sensor_bgr, cv2.COLOR_BGR2RGB)
    sensor_rgb = cv2.resize(sensor_rgb_big, (1000, 1000), interpolation=cv2.INTER_CUBIC)
    plt.imshow(sensor_rgb)
    plt.show()

    for sensor_i in sensor_names:
        df_temperature_si = df_temperature_norm[df_temperature_norm['sensor_name'] == sensor_i]
        df_temperature_si = df_temperature_si[df_temperature_si['sensor_spot'] < 7]
        # Generate_one_image_per_frame
        for f in df_temperature_si['frame_cum'].unique():
            df_temperature_temp = df_temperature_si[df_temperature_si['frame_cum'] == f]
            colors_dict_temp = colors_dict
            for sensor_spot in df_temperature_si['sensor_spot'].unique():
                row = df_temperature_temp[df_temperature_temp['sensor_spot'] == sensor_spot]
                temp_key = 'T' + str(sensor_spot + 1)
                r = int(row['r_channel'].to_numpy()[0])
                g = int(row['g_channel'].to_numpy()[0])
                b = int(row['b_channel'].to_numpy()[0])
                colors_dict_temp[temp_key] = [r, g, b]
                temperature_float = row['temperature'].to_numpy()[0]
                temperature_name = np.round(temperature_float, 2)
                dict_dataset['spot_' + str(sensor_spot) + '_r'].append(r)
                dict_dataset['spot_' + str(sensor_spot) + '_g'].append(g)
                dict_dataset['spot_' + str(sensor_spot) + '_b'].append(b)

            image_out, image_out_no_noise = assign_circle_colors(
                input_image=sensor_rgb,
                circles_df=circle_coords,
                colors_dict=colors_dict_temp,
                noise_perc=NO_NOISE_PERC,
            )
            image_name = str(sensor_i) + '_' + str(f) + '_' + str(temperature_name).replace(".", "-")
            # image_out_small = cv2.resize(image_out, (250, 250), interpolation=cv2.INTER_CUBIC)

            # SAVE IMAGE
            plt.imsave(save_img_path_noise + image_name + '.png', image_out)

            # SAVE IMAGE NO NOISE
            plt.imsave(save_img_path_without_noise + image_name + '.png', image_out_no_noise)

            # SAVE DICTIONARY
            dict_dataset['temperature_value'].append(temperature_float)
            dict_dataset['experiment_id'].append(experiment_name)
            dict_dataset['frame'].append(f)
            dict_dataset['sensor_name'].append(sensor_i)
            dict_dataset['image_name'].append(image_name)
            print(image_name)

    df_dataset = pd.DataFrame.from_dict(dict_dataset)
    return df_dataset


if __name__ == "__main__":
    # INPUTS
    PATH_DATAFRAME = '../../data/raw/df_experimental.csv'

    # Experiment NAME
    EXPERIMENT_NAME = 'experimental_temperature'

    # COLOR NORMALIZATION
    REF_SENSOR = 'sensor_1'

    date_string = today.strftime("%d_%m_%Y")

    INPUT_FOLDER_PATH = '../../data/raw/template/sensor_checker.png'
    SENSOR_COORDS = '../../data/raw/template/sensing_area_coords.csv'

    SAVE_FOLDER_PATH = '../../data/processed/ground_truths/generated_data_' + date_string + '/'
    SAVE_FOLDER_NO_NOISE_PATH = '../../data/processed/ground_truths/generated_no_noise_data_' + date_string + '/'

    if not os.path.exists(SAVE_FOLDER_PATH):
        os.makedirs(SAVE_FOLDER_PATH)
    if not os.path.exists(SAVE_FOLDER_NO_NOISE_PATH):
        os.makedirs(SAVE_FOLDER_NO_NOISE_PATH)

    SAVE_DF_PATH = '../../data/processed/ground_truths/'
    NAME_DATAFRAME = 'df_generated_data_' + date_string + '.csv'

    NO_NOISE_PERC = 0.98

    # ----------------------------------------------------------------------------------------------------
    # Sensor Template Data
    circle_coords_df = pd.read_csv(SENSOR_COORDS)

    remaining_areas = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
    colors_dict = {}
    for i in remaining_areas:
        colors_dict[i] = [0, 0, 0]

    dict_save_vals = ['temperature_value', 'experiment_id', 'sensor_name', 'image_name', 'frame']
    dict_dataset = {}
    for i in dict_save_vals:
        dict_dataset[i] = []

    # ----------------------------------------------------------------------------------------------------
    # GENERATE DATASET
    df_dataset = main_dataset_generation(
        experimental_df_path=PATH_DATAFRAME,
        path_sensor_image=INPUT_FOLDER_PATH,
        circle_coords=circle_coords_df,
        colors_dict=colors_dict,
        dict_dataset=dict_dataset,
        save_img_path_noise=SAVE_FOLDER_PATH,
        save_img_path_without_noise=SAVE_FOLDER_NO_NOISE_PATH,
        experiment_name=EXPERIMENT_NAME,
    )

    df_dataset.to_csv(SAVE_DF_PATH + NAME_DATAFRAME, index=False)
