# DCGAN for Anime Face Generation

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) in PyTorch to generate anime character faces.

## Project Structure

-   `config.py`: Contains all hyperparameters and configuration settings.
-   `models.py`: Defines the `Generator` and `Discriminator` network architectures.
-   `data_loader.py`: Handles downloading, preprocessing, and loading of the dataset.
-   `train.py`: The main script to train the DCGAN model.
-   `generate.py`: A script to generate images using a pre-trained generator model.
-   `requirements.txt`: A list of all the Python packages required to run the project.

## How to Run

### 1. Installation

Clone the repository and install the required packages:

```bash
git clone [https://your-repository-url.git](https://your-repository-url.git)
cd your-repository-name
pip install -r requirement.txt

2. Training
To train the model, run the train.py script:
        python train.py

3. Generate Images
To generate new images using the trained model, run the generate.py script:
        python generate.py


Note on Kaggle Authentication: The kagglehub library needs to authenticate to download the dataset. You might be prompted to log in or need to have your Kaggle API key (kaggle.json) set up in your user profile (C:\Users\<Your-Username>\.kaggle\kaggle.json).