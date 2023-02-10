import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


# Simple Denoising autoencoder with regression
def load_ae_regression_model_v0(
        input_shape_img,
        output_channels,
        n_filters=304,
        dense_neurons=1000,
):
    # Inputs
    input_image = layers.Input(shape=input_shape_img, name='input_image')

    # Denoising AE
    encoder = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same')(input_image)
    encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)
    encoder = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(encoder)
    encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)

    decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
    decoder = layers.UpSampling2D((2, 2))(decoder)
    decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(decoder)
    decoder = layers.UpSampling2D((2, 2))(decoder)

    output_decoder = layers.Conv2D(
        output_channels,
        kernel_size=(3, 3),
        activation='sigmoid',
        padding='same',
        name='output_decoder')(decoder)

    # Regression
    dense_reg = layers.Flatten()(output_decoder)
    dense_reg = layers.Dense(dense_neurons, activation='relu')(dense_reg)
    dense_reg = layers.Dense(int(dense_neurons / 2), activation='relu')(dense_reg)
    dense_reg = layers.Dense(int(dense_neurons / 4), activation='relu')(dense_reg)
    dense_reg = layers.Dense(int(dense_neurons / 8), activation='relu')(dense_reg)
    dense_reg = layers.Dense(int(dense_neurons / 16), activation='relu')(dense_reg)
    output_reg = layers.Dense(1, activation='linear', name='output_reg')(dense_reg)

    autoencoder_reg = Model([input_image], [output_decoder, output_reg])
    autoencoder_reg.summary()

    return autoencoder_reg


# Denoising autoencoder with mask as input and kernel as output, plus regression
def load_ae_regression_model(
        input_shape_img,
        input_shape_mask,
        output_channels,
        n_filters=304,
        dense_neurons=1000,
        cnn_depth=1,
):
    # Input Mask
    input_mask = layers.Input(shape=input_shape_mask, name='input_mask')

    # Input Img
    input_image = layers.Input(shape=input_shape_img, name='input_image')

    input_img_and_mask = tf.concat([input_image, input_mask], 3)

    encoder = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same')(
        input_img_and_mask)
    encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)

    for i in range(cnn_depth):
        n_filters = int(n_filters / 2)
        encoder = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(encoder)
        encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)

    decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
    decoder = layers.UpSampling2D((2, 2))(decoder)

    for j in range(cnn_depth):
        n_filters = n_filters * 2
        decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(decoder)
        decoder = layers.UpSampling2D((2, 2))(decoder)

    output_decoder = layers.Conv2D(output_channels, kernel_size=(3, 3), activation='sigmoid', padding='same',
                                   name='output_decoder')(decoder)

    # Mask the Input Before Regression
    # concat_mask = input_mask
    # for i in range(output_channels - 1):
    #     concat_mask = tf.concat([concat_mask, input_mask], 3)
    # output_with_mask = output_decoder * concat_mask

    # Concatenate Mask
    output_with_mask = tf.concat([output_decoder, input_mask], 3)

    print('output_with_mask', output_with_mask.shape)
    print('output_decoder', output_decoder.shape)

    # Regression
    dense_reg = layers.Flatten()(output_with_mask)
    dense_reg = layers.Dense(dense_neurons, activation='relu')(dense_reg)
    dense_reg = layers.Dense(int(dense_neurons / 2), activation='relu')(dense_reg)
    dense_reg = layers.Dense(int(dense_neurons / 4), activation='relu')(dense_reg)
    dense_reg = layers.Dense(int(dense_neurons / 8), activation='relu')(dense_reg)
    dense_reg = layers.Dense(int(dense_neurons / 16), activation='relu')(dense_reg)
    output_reg = layers.Dense(1, activation='linear', name='output_reg')(dense_reg)

    autoencoder_reg = Model(
        [input_image, input_mask],
        [output_decoder, output_reg]
    )
    autoencoder_reg.summary()
    return autoencoder_reg


# Denoising disentangled luminance and chrominance autoencoder with regression
def load_lum_chrom_regression_model(input_shape_lum, input_shape_chrom, input_shape_mask,
                                    output_channels_lum, output_channels_chrom,
                                    n_filters=304,
                                    dense_neurons=1000,
                                    cnn_depth=1,
                                    alpha=None,
                                    ):
    if alpha is None:
        alpha = [0.25, 0.25, 0.5]

    # Input Mask
    input_mask = layers.Input(shape=input_shape_mask, name='input_mask')

    # Luminance AE
    input_image_lum = layers.Input(shape=input_shape_lum, name='input_lum')
    # input_lum_mask = tf.concat([input_image_lum, input_mask],3)

    lum_encoder = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same')(
        input_image_lum)
    lum_encoder = layers.MaxPooling2D((2, 2), padding='same')(lum_encoder)

    for i in range(cnn_depth):
        lum_encoder = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(lum_encoder)
        lum_encoder = layers.MaxPooling2D((2, 2), padding='same')(lum_encoder)

    lum_decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(lum_encoder)
    lum_decoder = layers.UpSampling2D((2, 2))(lum_decoder)

    for j in range(cnn_depth):
        lum_decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(lum_decoder)
        lum_decoder = layers.UpSampling2D((2, 2))(lum_decoder)

    output_lum_decoder = layers.Conv2D(output_channels_lum, kernel_size=(3, 3), activation='sigmoid',
                                       padding='same',
                                       name='output_lum_decoder')(lum_decoder)

    image_corrected_lum = output_lum_decoder[:, :, :, 0:1]
    image_corrected_lum = tf.identity(image_corrected_lum, name='image_corrected_lum')

    # Chrominance AE
    input_image_chrom = layers.Input(shape=input_shape_chrom, name='input_chrom')

    chrom_encoder = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same')(
        input_image_chrom)
    chrom_encoder = layers.MaxPooling2D((2, 2), padding='same')(chrom_encoder)
    for i in range(cnn_depth):
        chrom_encoder = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(chrom_encoder)
        chrom_encoder = layers.MaxPooling2D((2, 2), padding='same')(chrom_encoder)

    chrom_decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(chrom_encoder)
    chrom_decoder = layers.UpSampling2D((2, 2))(chrom_decoder)

    for j in range(cnn_depth):
        chrom_decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(
            chrom_decoder)
        chrom_decoder = layers.UpSampling2D((2, 2))(chrom_decoder)

    output_chrom_decoder = layers.Conv2D(output_channels_chrom, kernel_size=(3, 3),
                                         activation='sigmoid', padding='same',
                                         name='output_chrom_decoder')(chrom_decoder)

    image_corrected_chrom = output_chrom_decoder[:, :, :, 0:2]
    image_corrected_chrom = tf.identity(image_corrected_chrom, name='image_corrected_chrom')

    # Input Before Regression
    integrated_output = tf.concat([image_corrected_lum, image_corrected_chrom, input_mask], 3)

    print(image_corrected_lum.shape)
    print(image_corrected_chrom.shape)
    print(integrated_output.shape)

    # Regression
    dense_reg = layers.Flatten()(integrated_output)
    dense_reg = layers.Dense(dense_neurons, activation='relu')(dense_reg)
    dense_reg = layers.Dense(int(dense_neurons / 2), activation='relu')(dense_reg)
    dense_reg = layers.Dense(int(dense_neurons / 4), activation='relu')(dense_reg)
    dense_reg = layers.Dense(int(dense_neurons / 8), activation='relu')(dense_reg)
    dense_reg = layers.Dense(int(dense_neurons / 16), activation='relu')(dense_reg)
    output_reg = layers.Dense(1, activation='linear', name='output_reg')(dense_reg)

    autoencoder_reg = Model(
        [input_image_lum, input_image_chrom, input_mask],
        [output_lum_decoder, output_chrom_decoder, output_reg]
    )
    autoencoder_reg.summary()
    return autoencoder_reg


# Denoising disentangled luminance and chrominance autoencoder with regression
def load_decoded_lum_chrom_reg_model(input_shape_img_ycrcb, input_shape_mask,
                                     output_channels_lum, output_channels_chrom,
                                     n_filters=304,
                                     dense_neurons=10000,
                                     cnn_depth=1,
                                     dense_layers=2,
                                     masked_input_reg=False,
                                     ):
    # Input Mask
    input_mask = layers.Input(shape=input_shape_mask, name='input_mask')

    # Input Image YCrCb
    input_image_ycrcb = layers.Input(shape=input_shape_img_ycrcb, name='input_image_ycrcb')

    integrated_input = tf.concat([input_image_ycrcb, input_mask], 3)

    # Joined Encoder
    encoder = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same')(integrated_input)
    encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)

    for i in range(cnn_depth):
        encoder = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(encoder)
        encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)

    # Luminance Decoder
    lum_decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
    lum_decoder = layers.UpSampling2D((2, 2))(lum_decoder)

    for j in range(cnn_depth):
        lum_decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(lum_decoder)
        lum_decoder = layers.UpSampling2D((2, 2))(lum_decoder)

    output_lum_decoder = layers.Conv2D(output_channels_lum,
                                       kernel_size=(3, 3),
                                       activation='sigmoid',
                                       padding='same',
                                       name='output_lum_decoder')(lum_decoder)

    # Chrominance Decoder
    chrom_decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
    chrom_decoder = layers.UpSampling2D((2, 2))(chrom_decoder)

    for j in range(cnn_depth):
        chrom_decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(chrom_decoder)
        chrom_decoder = layers.UpSampling2D((2, 2))(chrom_decoder)

    output_chrom_decoder = layers.Conv2D(output_channels_chrom,
                                         kernel_size=(3, 3),
                                         activation='sigmoid',
                                         padding='same',
                                         name='output_chrom_decoder')(chrom_decoder)

    # Light Kernel Decoder
    kernel_decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
    kernel_decoder = layers.UpSampling2D((2, 2))(kernel_decoder)

    for j in range(cnn_depth):
        kernel_decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(kernel_decoder)
        kernel_decoder = layers.UpSampling2D((2, 2))(kernel_decoder)

    output_kernel_decoder = layers.Conv2D(1,
                                          kernel_size=(3, 3),
                                          activation='sigmoid',
                                          padding='same',
                                          name='output_kernel_decoder')(kernel_decoder)

    # Input Before Regression
    if masked_input_reg:
        output_lum_decoder = output_lum_decoder * input_mask
        output_chrom_decoder = output_chrom_decoder * tf.concat([input_mask, input_mask], 3)
        output_kernel_decoder = output_kernel_decoder * tf.concat([input_mask, input_mask], 3)
        integrated_output = tf.concat([output_lum_decoder, output_chrom_decoder, output_kernel_decoder], 3)
    else:
        integrated_output = tf.concat([output_lum_decoder, output_chrom_decoder, output_kernel_decoder, input_mask], 3)

    print(integrated_output.shape)

    # Regression
    dense_reg = layers.Flatten()(integrated_output)
    dense_reg = layers.Dense(dense_neurons, activation='relu')(dense_reg)
    for k in range(dense_layers):
        dense_neurons = int(dense_neurons / 2)
        dense_reg = layers.Dense(dense_neurons, activation='relu')(dense_reg)
    output_reg = layers.Dense(1, activation='linear', name='output_reg')(dense_reg)

    autoencoder_reg = Model(
        [input_image_ycrcb, input_mask],
        [output_lum_decoder, output_chrom_decoder, output_kernel_decoder, output_reg]
    )
    autoencoder_reg.summary()
    return autoencoder_reg


def load_latent_lum_chrom_reg_model(input_shape_img_ycrcb, input_shape_mask,
                                    output_channels_lum, output_channels_chrom,
                                    n_filters=32,
                                    dense_neurons=10000,
                                    cnn_depth=1,
                                    dense_layers=4,
                                    ):
    # Input Mask
    input_mask = layers.Input(shape=input_shape_mask, name='input_mask')

    # Input Image YCrCb
    input_image_ycrcb = layers.Input(shape=input_shape_img_ycrcb, name='input_image_ycrcb')

    integrated_input = tf.concat([input_image_ycrcb, input_mask], 3)

    # Joined Encoder
    encoder = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same')(integrated_input)
    encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)

    for i in range(cnn_depth):
        n_filters = n_filters*2
        encoder = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(encoder)
        encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)

    # Luminance Decoder
    n_filters_decoder_lum = n_filters
    lum_decoder = layers.Conv2D(n_filters_decoder_lum, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
    lum_decoder = layers.UpSampling2D((2, 2))(lum_decoder)

    for j in range(cnn_depth):
        n_filters_decoder_lum = int(n_filters_decoder_lum / 2)
        lum_decoder = layers.Conv2D(n_filters_decoder_lum, kernel_size=(3, 3), activation='relu', padding='same')(lum_decoder)
        lum_decoder = layers.UpSampling2D((2, 2))(lum_decoder)

    output_lum_decoder = layers.Conv2D(output_channels_lum,
                                       kernel_size=(3, 3),
                                       activation='sigmoid',
                                       padding='same',
                                       name='output_lum_decoder')(lum_decoder)

    # Chrominance Decoder
    n_filters_decoder_chrom = n_filters
    chrom_decoder = layers.Conv2D(n_filters_decoder_chrom, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
    chrom_decoder = layers.UpSampling2D((2, 2))(chrom_decoder)

    for j in range(cnn_depth):
        n_filters_decoder_chrom = int(n_filters_decoder_chrom / 2)
        chrom_decoder = layers.Conv2D(n_filters_decoder_chrom, kernel_size=(3, 3), activation='relu', padding='same')(chrom_decoder)
        chrom_decoder = layers.UpSampling2D((2, 2))(chrom_decoder)

    output_chrom_decoder = layers.Conv2D(output_channels_chrom,
                                         kernel_size=(3, 3),
                                         activation='sigmoid',
                                         padding='same',
                                         name='output_chrom_decoder')(chrom_decoder)

    # Light Kernel Decoder
    n_filters_decoder_kernel = n_filters
    kernel_decoder = layers.Conv2D(n_filters_decoder_kernel, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
    kernel_decoder = layers.UpSampling2D((2, 2))(kernel_decoder)

    for j in range(cnn_depth):
        n_filters_decoder_kernel = int(n_filters_decoder_kernel / 2)
        kernel_decoder = layers.Conv2D(n_filters_decoder_kernel, kernel_size=(3, 3), activation='relu', padding='same')(kernel_decoder)
        kernel_decoder = layers.UpSampling2D((2, 2))(kernel_decoder)

    output_kernel_decoder = layers.Conv2D(1,
                                          kernel_size=(3, 3),
                                          activation='sigmoid',
                                          padding='same',
                                          name='output_kernel_decoder')(kernel_decoder)

    # Regression
    dense_reg = layers.Flatten()(encoder)
    dense_reg = layers.Dense(dense_neurons, activation='relu')(dense_reg)
    for k in range(dense_layers):
        dense_neurons = int(dense_neurons/2)
        dense_reg = layers.Dense(dense_neurons, activation='relu')(dense_reg)
    output_reg = layers.Dense(1, activation='linear', name='output_reg')(dense_reg)

    autoencoder_reg = Model(
        [input_image_ycrcb, input_mask],
        [output_lum_decoder, output_chrom_decoder, output_kernel_decoder, output_reg]
    )
    autoencoder_reg.summary()
    return autoencoder_reg


# ONLY REGRESSION ----------------------------------------------------------------------------------------------------
def load_regression_model(
        input_shape_img,
        input_shape_mask,
        n_filters=304,
        dense_neurons=1000,
        dense_layers=4,
        cnn_depth=1,
        with_cnn_encoder=False,
        masked_image=False,
        batch_norm=False,
):
    # Inputs
    input_image = layers.Input(shape=input_shape_img, name='input_image')

    # Input Mask
    input_mask = layers.Input(shape=input_shape_mask, name='input_mask')

    if masked_image:
        concat_mask = input_mask
        for i in range(2):
            concat_mask = tf.concat([concat_mask, input_mask], 3)
        integrated_output = input_image * concat_mask
    else:
        integrated_output = tf.concat([input_image, input_mask], 3)

    # CNN Module
    if with_cnn_encoder:
        encoder = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu',
                                padding='same')(integrated_output)
        encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)
        for i in range(cnn_depth):
            n_filters = int(n_filters / 2)
            encoder = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(encoder)
            encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)
    else:
        encoder = integrated_output

    # Fully Connected Network
    dense_reg = layers.Flatten()(encoder)
    dense_reg = layers.Dense(dense_neurons, activation='relu')(dense_reg)
    for k in range(dense_layers):
        if batch_norm:
            dense_reg = layers.BatchNormalization()(dense_reg)
        dense_neurons = int(dense_neurons / 2)
        dense_reg = layers.Dense(dense_neurons, activation='relu')(dense_reg)
    output_reg = layers.Dense(1, activation='linear', name='output_reg')(dense_reg)

    model_reg = Model([input_image, input_mask], [output_reg])
    model_reg.summary()
    return model_reg


# ONLY DENOISING ----------------------------------------------------------------------------------------------------
# Denoising with kernel
def load_denoising_model(input_shape_img_ycrcb, input_shape_mask,
                         output_channels=3,
                         n_filters=304,
                         cnn_depth=1,
                         ):
    # Input Mask
    input_mask = layers.Input(shape=input_shape_mask, name='input_mask')

    # Input Image YCrCb
    input_image_ycrcb = layers.Input(shape=input_shape_img_ycrcb, name='input_image_ycrcb')

    integrated_input = tf.concat([input_image_ycrcb, input_mask], 3)

    encoder = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), activation='relu', padding='same')(
        integrated_input)
    encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)

    for i in range(cnn_depth):
        n_filters = int(n_filters / 2)
        encoder = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(encoder)
        encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)

    decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(encoder)
    decoder = layers.UpSampling2D((2, 2))(decoder)

    for j in range(cnn_depth):
        n_filters = n_filters * 2
        decoder = layers.Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(decoder)
        decoder = layers.UpSampling2D((2, 2))(decoder)

    output_decoder = layers.Conv2D(output_channels, kernel_size=(3, 3), activation='sigmoid',
                                                padding='same', name='output_decoder')(decoder)

    autoencoder_denoise = Model(
        [input_image_ycrcb, input_mask],
        [output_decoder]
    )
    autoencoder_denoise.summary()
    return autoencoder_denoise
