import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from data_loader import load_imagenet_data
from moe_model import MoE
import numpy as np


class MoELoss(nn.Module):
    def __init__(self):
        super(MoELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output, target, gating_weights):
        mse_loss = self.criterion(output, target)

        # 增加门控网络的熵正则化，防止门控网络只使用一个专家
        entropy_loss = -torch.sum(gating_weights * torch.log(gating_weights + 1e-6))
        total_loss = mse_loss + 0.01 * entropy_loss
        return total_loss


def train_moe_model(num_epochs=10, batch_size=8, learning_rate=1e-3, noise_type='gaussian'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    train_loader, val_loader = load_imagenet_data(batch_size, noise_type)

    # 初始化 MoE 模型
    model = MoE(num_experts=3, n_channels=3, n_classes=3).to(device)
    criterion = MoELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, _ in tqdm(train_loader):
            inputs = inputs.to(device)
            noisy_inputs = inputs.clone()

            # 前向传播
            optimizer.zero_grad()
            outputs = model(noisy_inputs)

            # 计算损失
            gating_weights = model.gating_network(noisy_inputs)
            loss = criterion(outputs, inputs, gating_weights)

            # 反向传播
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
        validate_moe_model(model, val_loader, criterion, device)


def validate_moe_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            gating_weights = model.gating_network(inputs)
            loss = criterion(outputs, inputs, gating_weights)
            val_loss += loss.item()
    print(f"Validation Loss: {val_loss / len(val_loader)}")


if __name__ == "__main__":
    train_moe_model(num_epochs=10, batch_size=8, learning_rate=1e-3, noise_type='gaussian')
