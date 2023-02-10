import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

import os
import cv2
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

import wandb
from wandb.keras import WandbCallback

from sklearn.model_selection import train_test_split

from models import load_latent_lum_chrom_reg_model
from generators import get_generator, custom_decoded_lum_chrom_reg_generator


def objective(trial, custom_model, get_generator_function, custom_generator):
    # Image Parameters
    IMG_HEIGHT = 72
    IMG_WIDTH = 72
    INPUT_SHAPE_MASK = (IMG_HEIGHT, IMG_WIDTH, 1)
    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

    # Model Parameters
    MODEL_TYPE = 'chrom_lum_reg_lat'
    OUTPUT_CHANNELS_LUM = 1
    OUTPUT_CHANNELS_CHROM = 2
    LR = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    REG_LOSS = 'mse'

    LOSS_FX = trial.suggest_categorical(
        "loss_function",
        ['mse', 'binary_crossentropy', 'ssim']
    )

    CNN_DEPTH = trial.suggest_int("num_cnn_layers", 1, 2)
    CNN_FILTERS = trial.suggest_int("cnn_filters", 50, 100)
    DENSE_NEURONS = trial.suggest_int("dense_neurons", 1e3, 1e5, log=True)
    DENSE_LAYERS = trial.suggest_int("dense_layers", 4, 8)

    # Loss weights
    WEIGHT_LOSS_REG = 1
    WEIGHT_LOSS_LUM = trial.suggest_float("weight_loss_lum", 1e-3, 1e3, log=True)
    WEIGHT_LOSS_CHROM = trial.suggest_float("weight_loss_chrom", 1e-3, 1e3, log=True)
    WEIGHT_LOSS_KERNEL = trial.suggest_float("weight_loss_kernel", 1e-3, 1e3, log=True)

    # Training Parameters
    SEED = 8
    BATCH_SIZE = 8
    EPOCHS = 25
    PATIENCE_LR_PLATEAU = 5
    PATIENCE_EARLY_STOPPING = 5
    STEPS_PER_EPOCH = 30
    VALIDATION_STEPS = 15

    TRAIN_SENSORS = ['sensor_1']
    VAL_SENSORS = ['sensor_3']
    TEST_SENSORS = ['sensor_2']
    VAL_RATIO = 0.3

    NOISE_TYPE = ['polynomial', 'radial']
    OOD_NOISE_TYPE = ['polyrad']

    # PARAMETERS
    params = {
        "model_type": MODEL_TYPE,
        "weight_loss_reg": WEIGHT_LOSS_REG,
        "weight_loss_lum": WEIGHT_LOSS_LUM,
        "weight_loss_chrom": WEIGHT_LOSS_CHROM,
        "weight_loss_kernel": WEIGHT_LOSS_KERNEL,
        "learning_rate": LR,
        "lum_loss_function": LOSS_FX,
        "chrom_loss_function": LOSS_FX,
        "kernel_loss_function": LOSS_FX,
        "reg_loss_function": REG_LOSS,
        "cnn_filters": CNN_FILTERS,
        "dense_neurons": DENSE_NEURONS,
        "dense_layers": DENSE_LAYERS,
        "cnn_depth": CNN_DEPTH,
        "batch_size": BATCH_SIZE,
        "img_height": IMG_HEIGHT,
        "img_width": IMG_WIDTH,
        "seed": SEED,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "validation_steps": VALIDATION_STEPS,
        "train_sensors": TRAIN_SENSORS,
        "val_sensors": VAL_SENSORS,
        "test_sensors": TEST_SENSORS,
        "patience_lr_plateau": PATIENCE_LR_PLATEAU,
        "patience_early_stopping": PATIENCE_EARLY_STOPPING,
        "noise_distribution": NOISE_TYPE,
        "noise_out_of_distribution": OOD_NOISE_TYPE,
        "val_ratio": VAL_RATIO,
    }

    # WandB configuration
    PROJECT_NAME_WANDB = 'lck-reg-latent-opt-' + LOSS_FX
    run = wandb.init(
        project=PROJECT_NAME_WANDB,
        config=params,
    )
    config = wandb.config

    # -------------------------------------------------------------------------------------------
    # Load Model
    model_denoising_reg = custom_model(
        input_shape_img_ycrcb=INPUT_SHAPE,
        input_shape_mask=INPUT_SHAPE_MASK,
        output_channels_lum=OUTPUT_CHANNELS_LUM,
        output_channels_chrom=OUTPUT_CHANNELS_CHROM,
        n_filters=CNN_FILTERS,
        dense_neurons=DENSE_NEURONS,
        dense_layers=DENSE_LAYERS,
        cnn_depth=CNN_DEPTH,
    )

    # -------------------------------------------------------------------------------------------
    # Load Generators

    # Load dataframe
    df_generators = pd.read_csv(PATH_DF_DATASET)
    # Filter out-of-distribution-noise
    df_generators_filt = df_generators[df_generators['type_of_kernel'].isin(NOISE_TYPE)]

    # Train Val Split by ratio
    df_train_val = df_generators_filt[df_generators_filt['sensor_name'].isin([TRAIN_SENSORS[0], VAL_SENSORS[0]])]
    df_train, df_val = train_test_split(df_train_val, test_size=VAL_RATIO, random_state=2)

    # # Train val split by sensor
    # df_train = df_generators_filt[df_generators_filt['sensor_name'].isin(TRAIN_SENSORS)]
    # df_val = df_generators_filt[df_generators_filt['sensor_name'].isin(VAL_SENSORS)]

    dfs_train_val = [df_train, df_val]

    # Noisy Image Generators
    noise_generators_dict = {}
    noise_generators_names = [
        'noise_train_generator',
        'noise_val_generator',
    ]

    for i, val in enumerate(noise_generators_names):
        generator = get_generator_function(
            df_input=dfs_train_val[i],
            path_column='path_noise',
            labels_columns=['temperature_value'],
            batch_size=BATCH_SIZE,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
            seed=SEED,
            class_mode='other',
            color_mode='rgb',
            shuffle=True,
        )
        noise_generators_dict[val] = generator

    # GT Image Generators
    gt_generators_dict = {}
    gt_generators_names = [
        'gt_train_generator',
        'gt_val_generator',
    ]

    for i, val in enumerate(gt_generators_names):
        generator = get_generator_function(
            df_input=dfs_train_val[i],
            path_column='path_gt',
            labels_columns=['temperature_value'],
            batch_size=BATCH_SIZE,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
            seed=SEED,
            class_mode='other',
            color_mode='rgb',
            shuffle=True,
        )
        gt_generators_dict[val] = generator

    # Light Kernel Generators
    light_generators_dict = {}
    light_generators_names = [
        'light_kernel_train_generator',
        'light_kernel_val_generator',
    ]

    for i, val in enumerate(light_generators_names):
        generator = get_generator_function(
            df_input=dfs_train_val[i],
            path_column='path_light_kernel',
            labels_columns=['temperature_value'],
            batch_size=BATCH_SIZE,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
            seed=SEED,
            class_mode='other',
            color_mode='grayscale',
            shuffle=True,
        )
        light_generators_dict[val] = generator

    # Mask Batch
    sensor_bgr = cv2.imread(PATH_MASK)
    sensor_gray_big = cv2.cvtColor(sensor_bgr, cv2.COLOR_BGR2GRAY)
    sensor_gray = cv2.resize(sensor_gray_big, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
    mask_array = np.array(sensor_gray)
    mask_array = (mask_array < 50) * 1.0
    mask_batch = np.expand_dims(np.array([mask_array for i in range(BATCH_SIZE)]), axis=-1)

    generator_train = custom_generator(
        noise_gen=noise_generators_dict['noise_train_generator'],
        gt_gen=gt_generators_dict['gt_train_generator'],
        light_kernel_gen=light_generators_dict['light_kernel_train_generator'],
        mask=mask_batch,
    )
    generator_val = custom_generator(
        noise_gen=noise_generators_dict['noise_val_generator'],
        gt_gen=gt_generators_dict['gt_val_generator'],
        light_kernel_gen=light_generators_dict['light_kernel_val_generator'],
        mask=mask_batch,
    )

    # -----------------------------------------------------------------------------------------
    # Loss function
    def ssim_loss(y_true, y_pred):
        loss = tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
        abs_loss = tf.math.abs(
            loss, name=None
        )
        return abs_loss

    if LOSS_FX == 'ssim':
        LOSS_FX = ssim_loss

    # Compile Model
    model_denoising_reg.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=[LOSS_FX, LOSS_FX, LOSS_FX, REG_LOSS],
        loss_weights=[
            WEIGHT_LOSS_LUM,
            WEIGHT_LOSS_CHROM,
            WEIGHT_LOSS_KERNEL,
            WEIGHT_LOSS_REG,
        ],
    )

    # -----------------------------------------------------------------------------------------
    # Callbacks for keras and WandB
    callbacks = []

    # WandB callback
    logging_callback = WandbCallback(
        save_model=False,
    )
    callbacks.append(logging_callback)
    print('wandb', str(run.id), str(run.name))

    # Keras Callbacks
    if SAVE_MODEL:
        temp_path_save_model = PATH_SAVE_MODEL + str(run.id) + '_' + str(run.name)
        os.mkdir(temp_path_save_model)
        keras_save_model_callback = tf.keras.callbacks.ModelCheckpoint(
            save_best_only=True,
            save_weights_only=False,
            filepath=temp_path_save_model + '/' + MODEL_NAME,
        )
        callbacks.append(keras_save_model_callback)

    keras_reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=PATIENCE_LR_PLATEAU,
        min_lr=1e-7,
    )
    callbacks.append(keras_reduce_lr_callback)

    keras_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE_EARLY_STOPPING,
    )
    callbacks.append(keras_early_stopping)

    # -------------------------------------------------------------------------------------------
    # Train Model
    model_denoising_reg.fit(
        x=generator_train,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=generator_val,
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks,
    )
    # --------------------------------------------------------------------------------------------
    # Evaluate Model
    score_reg = []
    N_BATCHES_EVAL = 3

    for n in range(N_BATCHES_EVAL):
        val_batch_inputs, val_batch_outputs = next(generator_val)

        input_image_tensor = np.array(val_batch_inputs['input_image_ycrcb'])
        mask_tensor = np.array(val_batch_inputs['input_mask']).astype(float)

        # predictions
        predictions = model_denoising_reg.predict([
            input_image_tensor,
            mask_tensor
        ])

        # variable predictions
        temp_gt_reg = val_batch_outputs['output_reg']
        temp_pred_reg = predictions[3]
        print(temp_gt_reg)
        print(temp_pred_reg)

        temp_score = tf.keras.metrics.mean_squared_error(
            y_true=temp_gt_reg,
            y_pred=temp_pred_reg,
        )
        score_reg.append(temp_score)
        print('batch score' + str(N_BATCHES_EVAL), score_reg)

    # Average test score in N batches
    score_reg_mean = np.mean(np.array(score_reg))
    print('score_reg_mean', score_reg_mean)

    wandb.finish()
    tf.keras.backend.clear_session()

    return score_reg_mean


if __name__ == "__main__":

    # Paths Data
    PATH_DF_DATASET = 'data/processed/color_alterations/color_alterations_24_01_2023/df_noisy_data_24_01_2023.csv'
    PATH_MASK = 'data/raw/mask.png'
    SAVE_MODEL = False
    PATH_SAVE_MODEL = 'models/03_11_2022_ae_reg/'
    MODEL_NAME = 'model.{epoch:02d}.h5'
    N_TRIALS = 30

    OPTUNA_STUDY_NAME = 'denoising-lat-reg-v0'

    # ------------------------------------------------------------------------------------------
    # Create optuna study
    storage_name = "sqlite:///{}.db".format(OPTUNA_STUDY_NAME)
    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            custom_model=load_latent_lum_chrom_reg_model,
            get_generator_function=get_generator,
            custom_generator=custom_decoded_lum_chrom_reg_generator,
        ),
        n_trials=N_TRIALS
    )

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    df_optimization = study.trials_dataframe()  # attrs=("number", "value", "params", "state")
    df_optimization.to_csv('optuna_trials/' + OPTUNA_STUDY_NAME + '.csv')
