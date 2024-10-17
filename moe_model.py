import torch
import torch.nn as nn
from unet.unet_model import UNet


class MoE(nn.Module):
    def __init__(self, num_experts=3, n_channels=3, n_classes=3, bilinear=False):
        super(MoE, self).__init__()
        # 创建多个 UNet 专家
        self.experts = nn.ModuleList([UNet(n_channels, n_classes, bilinear) for _ in range(num_experts)])
        # 门控网络，用于为每个专家分配权重
        self.gating_network = nn.Sequential(
            nn.Conv2d(n_channels, num_experts, kernel_size=1),
            nn.Softmax(dim=1)  # 生成每个专家的权重
        )

    def forward(self, x):
        gating_weights = self.gating_network(x)  # [batch_size, num_experts, H, W]
        expert_outputs = [expert(x) for expert in self.experts]  # 每个专家的输出
        expert_outputs = torch.stack(expert_outputs, dim=1)  # 将专家输出堆叠 [batch_size, num_experts, n_classes, H, W]

        gating_weights = gating_weights.unsqueeze(2)  # [batch_size, num_experts, 1, H, W]
        output = torch.sum(gating_weights * expert_outputs, dim=1)  # 加权求和后的最终输出 [batch_size, n_classes, H, W]

        return output
