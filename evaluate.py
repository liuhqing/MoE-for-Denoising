import torch
import cv2
from moe_model import MoE
import numpy as np


def load_image(image_path, image_size=(128, 128)):
    """
    加载并预处理图像：调整大小并归一化。
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # 归一化到 [0, 1]
    return image


def reverse_normalization(tensor_image):
    """
    反归一化并将 tensor 转换为 numpy 格式，用于保存为图像文件。
    """
    tensor_image = tensor_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0  # 转换为 [0, 255]
    return np.clip(tensor_image, 0, 255).astype(np.uint8)


def predict(image_path, model_path):
    """
    使用训练好的 MoE 模型对带噪声的图像进行去噪处理。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载预训练的 MoE 模型
    model = MoE(num_experts=3, n_channels=3, n_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for prediction")
        model = nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    # 加载带噪声的图像
    image = load_image(image_path).to(device)

    # 预测
    with torch.no_grad():
        denoised_image = model(image)

    # 反归一化并保存去噪后的图像
    denoised_image = reverse_normalization(denoised_image)
    output_path = "denoised_output.jpg"
    cv2.imwrite(output_path, denoised_image)
    print(f"Denoised image saved to {output_path}")


if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Predict using MoE model for denoising')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the noisy image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')

    args = parser.parse_args()

    # 执行预测
    predict(args.image_path, args.model_path)
