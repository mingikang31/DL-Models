"""Guided Diffusion Model for Image Generation"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super(GaussianDiffusion, self).__init__()
        self.model = model
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Move tensors to the same device as the model
        device = next(model.parameters()).device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)

    def forward(self, x_0):
        noise = torch.randn_like(x_0)
        t = torch.randint(0, self.timesteps, (x_0.size(0),), device=x_0.device)
        alpha_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        noisy_x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
        return noisy_x_t, noise, t
    
    def sample(self, shape):
        device = next(self.model.parameters()).device
        x_t = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            alpha_t = self.alpha_cumprod[t]
            beta_t = self.betas[t]
            t_tensor = torch.tensor([t], device=device).expand(shape[0])
            noise_pred = self.model(x_t, t_tensor)
            x_t = (x_t - beta_t * noise_pred) / torch.sqrt(alpha_t)
            if t > 0:
                x_t += torch.sqrt(beta_t) * torch.randn_like(x_t)
        return x_t

    def loss(self, x_0):
        noisy_x_t, noise, t = self.forward(x_0)
        noise_pred = self.model(noisy_x_t, t)
        loss = F.mse_loss(noise_pred, noise)
        return loss

def visualize_samples(samples, nrow=8):
    # Ensure samples are in [0, 1] range for visualization
    samples = samples.clamp(0, 1)
    grid = make_grid(samples, nrow=nrow)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.show()

def train_diffusion_model(model, dataloader, epochs=10, lr=1e-4):
    diffusion = GaussianDiffusion(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            x_0 = batch[0].to(next(model.parameters()).device)
            optimizer.zero_grad()
            loss = diffusion.loss(x_0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')

    return diffusion
def generate_images(diffusion, num_samples=16, image_shape=(3, 64, 64)):
    diffusion.model.eval()
    with torch.no_grad():
        samples = diffusion.sample((num_samples, *image_shape))
        samples = (samples + 1) / 2  # Normalize to [0, 1]
        return samples.clamp(0, 1)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, t=None):
        x = F.relu(self.conv1(x))
        return self.conv2(x)
def main():

    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    

    # Instantiate the model
    model = SimpleModel().to('mps')

    # Train the diffusion model
    diffusion_model = train_diffusion_model(model, dataloader)

    # Generate images
    samples = generate_images(diffusion_model)
    visualize_samples(samples)

if __name__ == "__main__":
    main()