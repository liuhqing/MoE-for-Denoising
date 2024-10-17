import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from noise_functions import add_gaussian_noise, add_salt_and_pepper_noise, add_poisson_noise
import numpy as np


class AddNoiseTransform:
    """给图像添加不同噪声的 transform"""

    def __init__(self, noise_type='gaussian'):
        self.noise_type = noise_type

    def __call__(self, image):
        if self.noise_type == 'gaussian':
            return add_gaussian_noise(image)
        elif self.noise_type == 'salt_and_pepper':
            return add_salt_and_pepper_noise(image)
        elif self.noise_type == 'poisson':
            return add_poisson_noise(image)
        else:
            raise ValueError("Unsupported noise type")


def load_imagenet_data(batch_size, noise_type):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: np.array(img)),
        AddNoiseTransform(noise_type),
        transforms.ToTensor()  # 转换回 Tensor 格式
    ])

    train_dataset = datasets.ImageNet(root='./data', split='train', transform=transform, download=True)
    val_dataset = datasets.ImageNet(root='./data', split='val', transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
