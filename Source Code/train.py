# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

import config
from models import Generator, Discriminator
from data_loader import get_dataloader

def main():
    """
    Main training loop for the DCGAN.
    """
    # Get the dataloader
    dataloader = get_dataloader()

    # Initialize models
    netG = Generator(config.NZ, config.NGF, config.NC).to(config.DEVICE)
    netD = Discriminator(config.NC, config.NDF).to(config.DEVICE)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=config.LR, betas=(config.BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.LR, betas=(config.BETA1, 0.999))

    # Fixed noise for visualizing generator's progress
    fixed_noise = torch.randn(64, config.NZ, 1, 1, device=config.DEVICE)
    real_label = 1.
    fake_label = 0.

    # Lists to track progress
    img_list = []
    G_losses = []
    D_losses = []

    print("Starting Training Loop...")
    for epoch in range(config.NUM_EPOCHS):
        for i, data in enumerate(dataloader, 0):
            # (1) Update Discriminator network
            netD.zero_grad()
            real_cpu = data[0].to(config.DEVICE)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=config.DEVICE)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, config.NZ, 1, 1, device=config.DEVICE)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update Generator network
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print(
                    f'[{epoch}/{config.NUM_EPOCHS}][{i}/{len(dataloader)}]\t'
                    f'Loss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\t'
                    f'D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}'
                )

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (i % 500 == 0) or ((epoch == config.NUM_EPOCHS-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    print("Training finished.")

    # Save models
    torch.save(netG.state_dict(), config.GENERATOR_MODEL_PATH)
    torch.save(netD.state_dict(), config.DISCRIMINATOR_MODEL_PATH)

    # Plot results
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{config.OUTPUT_DIR}/loss_plot.png")
    plt.show()

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig(f"{config.OUTPUT_DIR}/fake_images.png")
    plt.show()

if __name__ == '__main__':
    main()