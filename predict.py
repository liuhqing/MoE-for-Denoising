import torch
import cv2
from moe_model import MoE
import numpy as np


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # 归一化
    return image


def reverse_normalization(tensor_image):
    tensor_image = tensor_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
    return np.clip(tensor_image, 0, 255).astype(np.uint8)


def predict(image_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MoE(num_experts=3, n_channels=3, n_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image = load_image(image_path).to(device)
    with torch.no_grad():
        denoised_image = model(image)

    denoised_image = reverse_normalization(denoised_image)
    cv2.imwrite("denoised_output.jpg", denoised_image)
    print("Denoised image saved to denoised_output.jpg")


if __name__ == "__main__":
    predict("path_to_noisy_image.jpg", "path_to_trained_model.pth")
