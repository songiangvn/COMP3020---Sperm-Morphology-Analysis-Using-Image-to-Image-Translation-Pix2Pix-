"""
Calculate quality metrics for cGAN inference results
"""
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from baselines.cgan import Generator
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(generated, ground_truth):
    """Calculate SSIM and PSNR"""
    gen_np = np.clip(np.array(generated), 0, 1)
    gt_np = np.clip(np.array(ground_truth), 0, 1)
    
    ssim_val = ssim(gen_np, gt_np, channel_axis=2, data_range=1.0)
    psnr_val = psnr(gt_np, gen_np, data_range=1.0)
    mae = np.mean(np.abs(gen_np - gt_np))
    
    return ssim_val, psnr_val, mae

def evaluate_cgan(checkpoint_path, data_dir, device='cuda:0'):
    """Evaluate cGAN on full test set"""
    # Load model
    model = Generator(in_channels=3, out_channels=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()
    
    # Get test PHASE files
    phase_files = sorted([f for f in os.listdir(data_dir) if 'PHASE' in f and f.endswith('.png')])
    n = len(phase_files)
    test_start = int(0.85 * n)
    test_files = phase_files[test_start:]
    
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
        for phase_file in test_files:
            # Load PHASE (input)
            phase_path = os.path.join(data_dir, phase_file)
            holography = Image.open(phase_path).convert('RGB')
            
            # Load STAIN (ground truth)
            stain_file = phase_file.replace('PHASE', 'STAIN')
            stain_path = os.path.join(data_dir, stain_file)
            stained_gt = Image.open(stain_path).convert('RGB')
            
            holo_tensor = transform(holography).unsqueeze(0).to(device)
            generated = model(holo_tensor)
            
            gen_np = denorm(generated[0]).cpu().permute(1, 2, 0).numpy()
            gt_np = np.array(stained_gt.resize((256, 256))) / 255.0
            
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
    checkpoint_path = './outputs_baselines/cgan/cgan_best.pth'
    data_dir = '/home/23giang.ns/ML_Project/virtual_stainning/both'
    
    print("="*60)
    print("cGAN Quality Metrics Evaluation")
    print("="*60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    results = evaluate_cgan(checkpoint_path, data_dir, device)
    
    print(f"\n{'Metric':<15} {'Mean':<12} {'Std':<12}")
    print("-"*40)
    print(f"{'SSIM':<15} {results['ssim']:<12.4f} {results['ssim_std']:<12.4f}")
    print(f"{'PSNR':<15} {results['psnr']:<12.4f} {results['psnr_std']:<12.4f}")
    print(f"{'MAE':<15} {results['mae']:<12.4f} {results['mae_std']:<12.4f}")
    print(f"\nTest samples: {results['n_samples']}")
    print("="*60)
