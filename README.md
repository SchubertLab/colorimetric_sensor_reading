# Colorimetric Sensor Reading
Methods for automated digital reading of colorimetric sensors in settings with perturbed illumination conditions and low image-resolutions.

<img width="1080" alt="Screenshot 2023-02-11 at 00 48 10" src="https://user-images.githubusercontent.com/39317465/218223105-eb67c77c-58dc-435f-8ab3-b8c391b2e11d.png">

## Introduction
> The proposed methodology aims to improve the accuracy of colorimetric sensor reading in altered illumination conditions.

> We implement image processing and deep-learning (DL) methods that correct for non-uniform illumination alterations and accurately read the target variable from the color response of the sensor.

## Repository
1. [Dataset Generation](./src/data/)

── 1.1 [Dataset Generation](./src/data/make_dataset.py)

── 1.2 [Dataset Noise Augmentations](./src/data/augment_images.py)


2. [Models and Pipeline Training](./src/models/)

── 2.1 [Model Zoo](./src/models/models.py "models.py")

── 2.2 [Image Generator Zoo](./src/models/generators.py "generators.py")

── 2.3 [Train Multi-task autoencoder with latent regression](./src/models/train_model_wandb_optuna_latent.py)

── 2.4 [Train MLP for color-based prediction](./src/models/train_model_wandb_optuna_reg.py)


## Usage
> - Clone the repository
> ```bash
> git clone 'git@github.com:SchubertLab/colorimetric_sensor_reading.git'
> ```
> - Install dependencies
> ```bash
> conda env create -f enviornment.yml
> ```

