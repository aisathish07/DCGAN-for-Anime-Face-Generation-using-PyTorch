# config.py

import torch

# --- Hyperparameters for the DCGAN model ---

# Size of the latent z vector (input to the generator)
NZ = 100

# Number of channels in the training images (RGB)
NC = 3

# Size of feature maps in the generator
NGF = 64

# Size of feature maps in the discriminator
NDF = 64

# Spatial size of training images (resized to this size)
IMAGE_SIZE = 64

# Batch size for training
BATCH_SIZE = 128

# Number of training epochs
NUM_EPOCHS = 50

# Learning rate for the optimizers
LR = 0.0002

# Beta1 hyperparameter for the Adam optimizer
BETA1 = 0.5

# Device to run the training on (GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the dataset
DATASET_PATH = "./dataset"

# Path to save the trained generator model
GENERATOR_MODEL_PATH = "generator.pth"

# Path to save the trained discriminator model
DISCRIMINATOR_MODEL_PATH = "discriminator.pth"

# Path to save generated images
OUTPUT_DIR = "./output"