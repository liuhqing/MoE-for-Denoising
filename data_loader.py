import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from noise_functions import add_gaussian_noise, add_salt_and_pepper_noise, add_poisson_noise
import numpy as np

class AddNoiseTransform:
    """
    根据噪声类型为图像添加噪声的 Transform
    """
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
            raise ValueError(f"Unsupported noise type: {self.noise_type}")

def load_imagenet_data(batch_size, num_train_images=1000, num_val_images=200, num_workers=4):
    """
    加载 ImageNet 数据集，并为每张图像添加指定的噪声，划分为不同噪声类型
    """
    transform_gaussian = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: np.array(img)),
        AddNoiseTransform('gaussian'),
        transforms.ToTensor()
    ])

    transform_salt_and_pepper = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: np.array(img)),
        AddNoiseTransform('salt_and_pepper'),
        transforms.ToTensor()
    ])

    transform_poisson = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: np.array(img)),
        AddNoiseTransform('poisson'),
        transforms.ToTensor()
    ])

    # 加载完整的 ImageNet 数据集
    train_dataset = datasets.ImageNet(root='./data', split='train', download=True)
    val_dataset = datasets.ImageNet(root='./data', split='val', download=True)

    # 将训练集随机分成三部分，分别添加不同的噪声
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    split = len(indices) // 3

    gaussian_indices = indices[:split]
    salt_and_pepper_indices = indices[split:2*split]
    poisson_indices = indices[2*split:]

    gaussian_subset = Subset(train_dataset, gaussian_indices)
    salt_and_pepper_subset = Subset(train_dataset, salt_and_pepper_indices)
    poisson_subset = Subset(train_dataset, poisson_indices)

    # 加载不同噪声类型的数据集
    gaussian_loader = DataLoader(gaussian_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    salt_and_pepper_loader = DataLoader(salt_and_pepper_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    poisson_loader = DataLoader(poisson_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return gaussian_loader, salt_and_pepper_loader, poisson_loader
