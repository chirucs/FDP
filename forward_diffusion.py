# scripts/forward_diffusion.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Create directories for data and results
data_path = "../data/MNIST"
results_path = "../results"
os.makedirs(data_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

# Parameters
T = 100  # Number of time steps for noise addition
beta_start = 0.0001
beta_end = 0.02
betas = np.linspace(beta_start, beta_end, T)

# Prepare the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
data_loader = DataLoader(mnist_data, batch_size=1, shuffle=True)

# Sample a single image
sample_image, _ = next(iter(data_loader))
sample_image = sample_image.squeeze(0)  # Remove batch dimension

# Function for forward diffusion
def forward_diffusion(image, T, betas):
    noisy_images = []
    for t in range(T):
        noise = torch.randn_like(image)  # Generate Gaussian noise
        # Apply the forward diffusion formula
        image = (1 - betas[t])**0.5 * image + betas[t]**0.5 * noise
        noisy_images.append(image)
    return noisy_images

# Apply forward diffusion
noisy_images = forward_diffusion(sample_image, T, betas)

# Save images at specific intervals to the results folder
interval = T // 10  # Save 10 images spaced evenly through the process
for i in range(0, T, interval):
    img = noisy_images[i].numpy().squeeze()
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.savefig(os.path.join(results_path, f"noisy_image_step_{i}.png"))

print("Forward diffusion images saved to the 'results' folder.")
