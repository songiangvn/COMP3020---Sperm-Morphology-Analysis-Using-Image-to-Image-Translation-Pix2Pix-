"""
Calculate quality metrics for VAE inference results
"""
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from baselines.vae import VAE
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F

def calculate_metrics(generated, ground_truth):
    """Calculate SSIM and PSNR"""
    # Convert to numpy arrays [0, 1]
    if isinstance(generated, torch.Tensor):
        gen_np = generated.cpu().numpy()
    else:
        gen_np = np.array(generated)
    
    if isinstance(ground_truth, torch.Tensor):
        gt_np = ground_truth.cpu().numpy()
    else:
        gt_np = np.array(ground_truth)
    
    # Ensure range [0, 1]
    gen_np = np.clip(gen_np, 0, 1)
    gt_np = np.clip(gt_np, 0, 1)
    
    # Calculate SSIM (multi-channel)
    ssim_val = ssim(gen_np, gt_np, channel_axis=2, data_range=1.0)
    
    # Calculate PSNR
    psnr_val = psnr(gt_np, gen_np, data_range=1.0)
    
    # Calculate MAE
    mae = np.mean(np.abs(gen_np - gt_np))
    
    return ssim_val, psnr_val, mae

def evaluate_vae(checkpoint_path, data_dir, device='cuda:0'):
    """Evaluate VAE on full test set"""
    # Load model
    model = VAE(in_channels=3, out_channels=3, latent_dim=128).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get test images
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
    n = len(all_files)
    test_start = int(0.85 * n)
    test_files = all_files[test_start:]
    
    print(f"Evaluating on {len(test_files)} test images...")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    def denorm(x):
        return torch.clamp(x * 0.5 + 0.5, 0, 1)
    
    ssim_scores = []
    psnr_scores = []
    mae_scores = []
    
    with torch.no_grad():
        for img_file in test_files:
            img_path = os.path.join(data_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            
            w, h = img.size
            holography = img.crop((0, 0, w//2, h))
            stained_gt = img.crop((w//2, 0, w, h))
            
            # Transform
            holo_tensor = transform(holography).unsqueeze(0).to(device)
            
            # Inference
            recon, mu, logvar = model(holo_tensor)
            
            # Convert to numpy
            gen_np = denorm(recon[0]).cpu().permute(1, 2, 0).numpy()
            gt_np = np.array(stained_gt.resize((256, 256))) / 255.0
            
            # Calculate metrics
            ssim_val, psnr_val, mae = calculate_metrics(gen_np, gt_np)
            
            ssim_scores.append(ssim_val)
            psnr_scores.append(psnr_val)
            mae_scores.append(mae)
    
    return {
        'ssim': np.mean(ssim_scores),
        'ssim_std': np.std(ssim_scores),
        'psnr': np.mean(psnr_scores),
        'psnr_std': np.std(psnr_scores),
        'mae': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'n_samples': len(test_files)
    }

if __name__ == '__main__':
    checkpoint_path = './outputs_baselines/vae/vae_best.pth'
    data_dir = '/home/23giang.ns/ML_Project/virtual_stainning/both'
    
    print("="*60)
    print("VAE Quality Metrics Evaluation")
    print("="*60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    results = evaluate_vae(checkpoint_path, data_dir, device)
    
    print(f"\n{'Metric':<15} {'Mean':<12} {'Std':<12}")
    print("-"*40)
    print(f"{'SSIM':<15} {results['ssim']:<12.4f} {results['ssim_std']:<12.4f}")
    print(f"{'PSNR':<15} {results['psnr']:<12.4f} {results['psnr_std']:<12.4f}")
    print(f"{'MAE':<15} {results['mae']:<12.4f} {results['mae_std']:<12.4f}")
    print(f"\nTest samples: {results['n_samples']}")
    print("="*60)
