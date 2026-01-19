"""
Quick inference script for VAE to check results
"""
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from baselines.vae import VAE
import matplotlib.pyplot as plt

def load_vae_model(checkpoint_path, device):
    """Load trained VAE model"""
    model = VAE(in_channels=3, out_channels=3, latent_dim=128).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded VAE from epoch {checkpoint['epoch']+1}")
    print(f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    return model

def inference_samples(model, data_dir, num_samples=5, device='cuda:0'):
    """Inference on test samples"""
    # Get test PHASE files (last 15% of dataset)
    phase_files = sorted([f for f in os.listdir(data_dir) if 'PHASE' in f and f.endswith('.png')])
    n = len(phase_files)
    test_start = int(0.85 * n)
    test_files = phase_files[test_start:test_start+num_samples]
    
    print(f"\nInferencing {len(test_files)} test samples...")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Denormalize function
    def denorm(x):
        x = x * 0.5 + 0.5
        return torch.clamp(x, 0, 1)
    
    results = []
    
    with torch.no_grad():
        for idx, phase_file in enumerate(test_files):
            # Load PHASE (input)
            phase_path = os.path.join(data_dir, phase_file)
            holography = Image.open(phase_path).convert('RGB')
            
            # Load STAIN (ground truth)
            stain_file = phase_file.replace('PHASE', 'STAIN')
            stain_path = os.path.join(data_dir, stain_file)
            stained_gt = Image.open(stain_path).convert('RGB')
            
            # Transform
            holo_tensor = transform(holography).unsqueeze(0).to(device)
            
            # Inference
            recon, mu, logvar = model(holo_tensor)
            
            # Convert to numpy
            holo_np = denorm(holo_tensor[0]).cpu().permute(1, 2, 0).numpy()
            recon_np = denorm(recon[0]).cpu().permute(1, 2, 0).numpy()
            stained_np = np.array(stained_gt.resize((256, 256))) / 255.0
            
            results.append({
                'filename': phase_file,
                'holography': holo_np,
                'generated': recon_np,
                'ground_truth': stained_np
            })
            
            print(f"  [{idx+1}/{len(test_files)}] {phase_file}")
    
    return results

def visualize_results(results, output_dir='./inference_vae'):
    """Visualize and save results"""
    os.makedirs(output_dir, exist_ok=True)
    
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Holography input
        axes[i, 0].imshow(result['holography'])
        axes[i, 0].set_title(f"Input\n{result['filename']}")
        axes[i, 0].axis('off')
        
        # VAE generated
        axes[i, 1].imshow(result['generated'])
        axes[i, 1].set_title('VAE Generated')
        axes[i, 1].axis('off')
        
        # Ground truth
        axes[i, 2].imshow(result['ground_truth'])
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'vae_samples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Save individual images
    for i, result in enumerate(results):
        img_concat = np.concatenate([
            result['holography'],
            result['generated'],
            result['ground_truth']
        ], axis=1)
        
        img_pil = Image.fromarray((img_concat * 255).astype(np.uint8))
        img_path = os.path.join(output_dir, f'sample_{i+1}_{result["filename"]}')
        img_pil.save(img_path)
    
    print(f"✓ Individual images saved to: {output_dir}/")

if __name__ == '__main__':
    # Configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # GPU:5 mapped to cuda:0
    checkpoint_path = './outputs_baselines/vae/vae_best.pth'
    data_dir = '/home/23giang.ns/ML_Project/virtual_stainning/both'
    num_samples = 5
    
    print("="*60)
    print("VAE Inference - Quick Check")
    print("="*60)
    
    # Load model
    model = load_vae_model(checkpoint_path, device)
    
    # Inference
    results = inference_samples(model, data_dir, num_samples, device)
    
    # Visualize
    visualize_results(results)
    
    print("\n" + "="*60)
    print("Inference completed!")
    print("="*60)
