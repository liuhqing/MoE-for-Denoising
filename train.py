import argparse
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from data_loader import load_imagenet_data
from moe_model import MoE

# 解析命令行参数
parser = argparse.ArgumentParser(description='Train MoE for Image Denoising')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
args = parser.parse_args()


# 定义损失函数
class MoELoss(nn.Module):
    def __init__(self):
        super(MoELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output, target, gating_weights):
        mse_loss = self.criterion(output, target)

        # 增加门控网络的熵正则化，防止门控网络只使用一个专家
        entropy_loss = -torch.sum(gating_weights * torch.log(gating_weights + 1e-6))
        total_loss = mse_loss + 0.01 * entropy_loss  # 使用 0.01 作为正则化系数
        return total_loss


# 训练函数
def train_moe_model(model, gaussian_loader, salt_and_pepper_loader, poisson_loader, num_epochs, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    # 定义优化器和损失函数
    criterion = MoELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for (gaussian_data, _), (sp_data, _), (poisson_data, _) in zip(gaussian_loader, salt_and_pepper_loader,
                                                                       poisson_loader):
            gaussian_data = gaussian_data.to(device)
            sp_data = sp_data.to(device)
            poisson_data = poisson_data.to(device)

            # Forward pass for Gaussian noise
            optimizer.zero_grad()
            outputs = model(gaussian_data)
            gating_weights = model.module.gating_network(
                gaussian_data) if torch.cuda.device_count() > 1 else model.gating_network(gaussian_data)
            loss = criterion(outputs, gaussian_data, gating_weights)
            loss.backward()
            optimizer.step()

            # Forward pass for Salt and Pepper noise
            optimizer.zero_grad()
            outputs = model(sp_data)
            gating_weights = model.module.gating_network(
                sp_data) if torch.cuda.device_count() > 1 else model.gating_network(sp_data)
            loss = criterion(outputs, sp_data, gating_weights)
            loss.backward()
            optimizer.step()

            # Forward pass for Poisson noise
            optimizer.zero_grad()
            outputs = model(poisson_data)
            gating_weights = model.module.gating_network(
                poisson_data) if torch.cuda.device_count() > 1 else model.gating_network(poisson_data)
            loss = criterion(outputs, poisson_data, gating_weights)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(gaussian_loader):.4f}")
        scheduler.step(running_loss)


# 加载数据集并添加噪声
gaussian_loader, salt_and_pepper_loader, poisson_loader = load_imagenet_data(batch_size=args.batch_size)

# 初始化并训练模型
model = MoE(num_experts=3, n_channels=3, n_classes=3)
train_moe_model(model, gaussian_loader, salt_and_pepper_loader, poisson_loader, args.num_epochs, args.learning_rate)
