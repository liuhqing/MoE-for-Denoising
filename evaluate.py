import torch
from data_loader import load_imagenet_data
from moe_model import MoE
import torch.nn.functional as F
from tqdm import tqdm

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    psnr_sum = 0.0
    with torch.no_grad():
        for inputs, _ in tqdm(val_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)

            mse_loss = F.mse_loss(outputs, inputs)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_loss))
            psnr_sum += psnr.item()

            total_loss += mse_loss.item()

    avg_psnr = psnr_sum / len(val_loader)
    print(f"Avg PSNR: {avg_psnr:.4f} dB")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MoE(num_experts=3, n_channels=3, n_classes=3)
    model.load_state_dict(torch.load("path_to_trained_model.pth", map_location=device))
    val_loader = load_imagenet_data(batch_size=8, noise_type='gaussian')[1]
    evaluate(model, val_loader, device)
