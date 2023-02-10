import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_generator(df_input, path_column,
                  labels_columns, batch_size,
                  img_height, img_width, seed, class_mode,
                  color_mode='rgb', shuffle=True):
    images_gen = ImageDataGenerator(
        rescale=1. / 255,
    )

    generator = images_gen.flow_from_dataframe(
        df_input,
        x_col=path_column,
        y_col=labels_columns,
        seed=seed,
        shuffle=shuffle,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode=color_mode,
    )
    return generator


def custom_img_reg_generator(noise_gen, gt_gen, light_kernel_gen, mask, offset_yuv=0.5):
    while True:
        noise_batch = next(noise_gen)
        ground_truth_batch = next(gt_gen)
        light_kernel_batch = next(light_kernel_gen)

        # Inputs
        x_noise = noise_batch[0]
        x_ground_truth = ground_truth_batch[0]
        x_light_kernel = light_kernel_batch[0]

        # Outputs
        x_gt_light_kernel = tf.concat([x_ground_truth, x_light_kernel], 3)
        x_reg_labels = noise_batch[1]

        # # Concatenate in YCrCbSpace
        # x_noise_yuv = tf.image.rgb_to_yuv(x_noise)
        # x_noise_yuv_norm = tf.math.divide(tf.add(x_noise_yuv, offset_yuv), (1 + offset_yuv))
        # x_gt_yuv = tf.image.rgb_to_yuv(x_ground_truth)
        # x_gt_yuv_norm = tf.math.divide(tf.add(x_gt_yuv, offset_yuv), (1 + offset_yuv))

        if x_noise.shape[0] == noise_gen.batch_size:
            x_inputs = {
                'input_image': x_noise,
                'input_mask': mask,
            }

            x_outputs = {
                'output_decoder': x_gt_light_kernel,
                'output_reg': x_reg_labels,
            }

            yield x_inputs, x_outputs


def custom_lum_chrom_reg_generator(noise_gen, gt_gen, light_kernel_gen, mask):
    while True:
        noise_batch = next(noise_gen)
        ground_truth_batch = next(gt_gen)
        light_kernel_batch = next(light_kernel_gen)

        x_noise = noise_batch[0]
        x_ground_truth = ground_truth_batch[0]
        x_light_kernel = light_kernel_batch[0]

        x_noise_yuv = tf.image.rgb_to_yuv(x_noise)
        x_noise_lum = x_noise_yuv[:, :, :, 0:1]
        x_noise_chrom = x_noise_yuv[:, :, :, 1:3]

        x_gt_yuv = tf.image.rgb_to_yuv(x_ground_truth)
        x_gt_lum = x_gt_yuv[:, :, :, 0:1]
        x_gt_chrom = x_gt_yuv[:, :, :, 1:3]

        x_gt_lum_light_kernel = tf.concat([x_gt_lum, x_light_kernel], 3)
        x_gt_chrom_light_kernel = tf.concat([x_gt_chrom, x_light_kernel], 3)

        # diff_lum = x_gt_lum - x_noise_lum
        # diff_chrom = x_gt_chrom - x_noise_chrom

        x_reg_labels = noise_batch[1]

        if (x_noise.shape[0] == noise_gen.batch_size):
            x_inputs = {
                'input_lum': x_noise_lum,
                'input_chrom': x_noise_chrom,
                'input_mask': mask,
            }

            x_outputs = {
                'output_lum_decoder': x_gt_lum_light_kernel,
                'output_chrom_decoder': x_gt_chrom_light_kernel,
                'output_reg': x_reg_labels,
            }

            yield x_inputs, x_outputs


def custom_decoded_lum_chrom_reg_generator(noise_gen, gt_gen, light_kernel_gen, mask):
    while True:
        noise_batch = next(noise_gen)
        ground_truth_batch = next(gt_gen)
        light_kernel_batch = next(light_kernel_gen)

        x_noise = noise_batch[0]
        x_ground_truth = ground_truth_batch[0]
        x_light_kernel = light_kernel_batch[0]

        x_noise_yuv = tf.image.rgb_to_yuv(x_noise)
        x_noise_yuv_clip = tf.clip_by_value(x_noise_yuv, 0.0, 1.0)
        # x_noise_yuv_norm = tf.math.divide(tf.add(x_noise_yuv, 0.5), (1 + 0.5))

        x_gt_yuv = tf.image.rgb_to_yuv(x_ground_truth)
        x_gt_yuv_clip = tf.clip_by_value(x_gt_yuv, 0.0, 1.0)
        x_gt_lum_clip = x_gt_yuv_clip[:, :, :, 0:1]
        x_gt_chrom_clip = x_gt_yuv_clip[:, :, :, 1:3]

        x_reg_labels = noise_batch[1]

        if x_noise.shape[0] == noise_gen.batch_size:
            x_inputs = {
                'input_image_ycrcb': x_noise_yuv_clip,
                'input_mask': mask,
            }

            x_outputs = {
                'output_lum_decoder': x_gt_lum_clip,
                'output_chrom_decoder': x_gt_chrom_clip,
                'output_kernel_decoder': x_light_kernel,
                'output_reg': x_reg_labels,
            }

            yield x_inputs, x_outputs


def custom_img_reg_generator_v0(noise_gen, gt_gen):
    while True:
        noise_batch = next(noise_gen)
        ground_truth_batch = next(gt_gen)

        # Inputs
        x_noise = noise_batch[0]
        x_ground_truth = ground_truth_batch[0]

        # Outputs
        x_reg_labels = noise_batch[1]

        if x_noise.shape[0] == noise_gen.batch_size:
            x_inputs = {
                'input_image': x_noise,
            }

            x_outputs = {
                'output_decoder': x_ground_truth,
                'output_reg': x_reg_labels,
            }

            yield x_inputs, x_outputs


def custom_only_reg_generator(gt_gen, mask):
    while True:
        ground_truth_batch = next(gt_gen)
        # Inputs
        x_ground_truth = ground_truth_batch[0]

        # Outputs
        x_reg_labels = ground_truth_batch[1]

        if x_ground_truth.shape[0] == gt_gen.batch_size:
            x_inputs = {
                'input_image': x_ground_truth,
                'input_mask': mask,
            }

            x_outputs = {
                'output_reg': x_reg_labels,
            }

            yield x_inputs, x_outputs


def custom_denoising_generator(noise_gen, gt_gen, mask):
    while True:
        noise_batch = next(noise_gen)
        ground_truth_batch = next(gt_gen)

        x_noise = noise_batch[0]
        x_ground_truth = ground_truth_batch[0]

        x_noise_yuv = tf.image.rgb_to_yuv(x_noise)
        x_noise_yuv_clip = tf.clip_by_value(x_noise_yuv, 0.0, 1.0)

        x_gt_yuv = tf.image.rgb_to_yuv(x_ground_truth)
        x_gt_yuv_clip = tf.clip_by_value(x_gt_yuv, 0.0, 1.0)

        if x_noise.shape[0] == noise_gen.batch_size:
            x_inputs = {
                'input_image_ycrcb': x_noise_yuv_clip,
                'input_mask': mask,
            }

            x_outputs = {
                'output_decoder': x_gt_yuv_clip,
            }

            yield x_inputs, x_outputs