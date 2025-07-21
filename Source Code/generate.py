# generate.py

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import os

import config
from models import Generator

def generate_images(num_images=64):
    """
    Generates images using the trained generator model.
    """
    if not os.path.exists(config.GENERATOR_MODEL_PATH):
        print(f"Generator model not found at {config.GENERATOR_MODEL_PATH}. Please train the model first.")
        return

    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    # Load the trained generator model
    netG = Generator(config.NZ, config.NGF, config.NC).to(config.DEVICE)
    netG.load_state_dict(torch.load(config.GENERATOR_MODEL_PATH, map_location=config.DEVICE))
    netG.eval()

    # Generate images from random noise
    noise = torch.randn(num_images, config.NZ, 1, 1, device=config.DEVICE)
    with torch.no_grad():
        fake_images = netG(noise).detach().cpu()

    # Save and display the generated images
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Anime Faces")
    plt.imshow(np.transpose(vutils.make_grid(fake_images, padding=2, normalize=True), (1, 2, 0)))
    plt.savefig(f"{config.OUTPUT_DIR}/generated_images.png")
    plt.show()

    print(f"Generated images saved to {config.OUTPUT_DIR}/generated_images.png")

if __name__ == '__main__':
    generate_images()