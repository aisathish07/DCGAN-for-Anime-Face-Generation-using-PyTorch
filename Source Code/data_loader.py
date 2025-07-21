# data_loader.py

import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import kagglehub
import config

def get_dataloader():
    """
    Downloads the dataset and returns a DataLoader.
    """
    # Download the dataset
    path = kagglehub.dataset_download("splcher/animefacedataset")
    print("Path to dataset files:", path)

    # Create the dataset
    dataset = dset.ImageFolder(root=path,
                               transform=transforms.Compose([
                                   transforms.Resize(config.IMAGE_SIZE),
                                   transforms.CenterCrop(config.IMAGE_SIZE),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE,
                                             shuffle=True, num_workers=2)

    return dataloader