"""
Training utilities and losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """VGG-based Perceptual Loss for better texture learning with small datasets"""
    
    def __init__(self):
        super().__init__()
        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=True).features
        
        # Extract feature layers
        self.slice1 = nn.Sequential(*list(vgg)[:9])   # relu2_2
        self.slice2 = nn.Sequential(*list(vgg)[9:18]) # relu3_4
        self.slice3 = nn.Sequential(*list(vgg)[18:27]) # relu4_4
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Set to eval mode
        self.eval()
    
    def forward(self, x, y):
        """
        Compute perceptual loss between x and y
        
        Args:
            x: Generated image [-1, 1]
            y: Target image [-1, 1]
        """
        # Normalize to ImageNet stats (VGG expects [0, 1] then normalized)
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        y = (y + 1) / 2
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        x = (x - mean) / std
        y = (y - mean) / std
        
        # Extract features at multiple layers
        x1, y1 = self.slice1(x), self.slice1(y)
        x2, y2 = self.slice2(x1), self.slice2(y1)
        x3, y3 = self.slice3(x2), self.slice3(y2)
        
        # Compute L1 loss at each layer
        loss = F.l1_loss(x1, y1) + F.l1_loss(x2, y2) + F.l1_loss(x3, y3)
        return loss


class GANLoss(nn.Module):
    """GAN Loss with MSE (LSGAN style) or BCE"""
    
    def __init__(self, use_lsgan: bool = True):
        super().__init__()
        self.use_lsgan = use_lsgan
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
    
    def __call__(self, prediction, target_is_real):
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        return self.loss(prediction, target)


class Pix2PixLoss:
    """Combined loss for Pix2Pix with optional Perceptual Loss"""
    
    def __init__(self, lambda_l1: float = 100.0, lambda_perceptual: float = 10.0, 
                 use_lsgan: bool = True, use_perceptual: bool = True):
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.use_perceptual = use_perceptual
        
        self.gan_loss = GANLoss(use_lsgan=use_lsgan)
        self.l1_loss = nn.L1Loss()
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def generator_loss(self, fake_pred, fake_img, real_img):
        """
        Generator loss = GAN loss + L1 loss + Perceptual loss
        
        Args:
            fake_pred: Discriminator prediction on fake images
            fake_img: Generated images
            real_img: Real target images
        """
        # GAN loss (fool discriminator)
        gan_loss = self.gan_loss(fake_pred, target_is_real=True)
        
        # L1 loss (pixel-wise reconstruction)
        l1_loss = self.l1_loss(fake_img, real_img)
        
        # Perceptual loss (feature-level matching)
        if self.use_perceptual:
            perceptual_loss = self.perceptual_loss(fake_img, real_img)
            total_loss = gan_loss + self.lambda_l1 * l1_loss + self.lambda_perceptual * perceptual_loss
            return total_loss, gan_loss, l1_loss, perceptual_loss
        else:
            total_loss = gan_loss + self.lambda_l1 * l1_loss
            return total_loss, gan_loss, l1_loss, torch.tensor(0.0)
    
    def discriminator_loss(self, real_pred, fake_pred):
        """
        Discriminator loss = Real loss + Fake loss
        
        Args:
            real_pred: Discriminator prediction on real images
            fake_pred: Discriminator prediction on fake images
        """
        real_loss = self.gan_loss(real_pred, target_is_real=True)
        fake_loss = self.gan_loss(fake_pred, target_is_real=False)
        
        total_loss = (real_loss + fake_loss) * 0.5
        
        return total_loss, real_loss, fake_loss


class VAELoss:
    """VAE loss with reconstruction and KL divergence"""
    
    def __init__(self, kld_weight: float = 0.00025):
        self.kld_weight = kld_weight
        self.recon_loss = nn.L1Loss()  # or nn.MSELoss()
    
    def __call__(self, recon, target, mu, logvar):
        """
        VAE loss = Reconstruction loss + KL divergence
        
        Args:
            recon: Reconstructed images
            target: Target images
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Reconstruction loss
        recon_loss = self.recon_loss(recon, target)
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = kld_loss / (target.size(0) * target.size(1) * target.size(2) * target.size(3))
        
        # Total loss
        total_loss = recon_loss + self.kld_weight * kld_loss
        
        return total_loss, recon_loss, kld_loss


class SimpleLoss:
    """Simple L1 loss for U-Net"""
    
    def __init__(self):
        self.loss = nn.L1Loss()
    
    def __call__(self, pred, target):
        return self.loss(pred, target)


def get_loss_function(model_name: str, **kwargs):
    """Factory function to get loss function"""
    if model_name in ['pix2pix_modified', 'cgan']:
        return Pix2PixLoss(**kwargs)
    elif model_name == 'vae':
        return VAELoss(**kwargs)
    elif model_name == 'unet':
        return SimpleLoss()
    else:
        raise ValueError(f"Unknown model: {model_name}")


# Evaluation metrics

def calculate_psnr(img1, img2, max_val: float = 1.0):
    """Calculate PSNR between two images"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, window_size: int = 11, max_val: float = 1.0):
    """
    Calculate SSIM between two images
    Simplified version - for accurate SSIM, use pytorch_msssim library
    """
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return ssim_map.mean().item()


class MetricsCalculator:
    """Calculate multiple metrics"""
    
    def __init__(self):
        pass
    
    def calculate(self, pred, target):
        """
        Calculate all metrics
        
        Args:
            pred: Predicted images (B, C, H, W) in range [-1, 1]
            target: Target images (B, C, H, W) in range [-1, 1]
        
        Returns:
            dict of metrics
        """
        # Convert to [0, 1] range
        pred_01 = (pred + 1) / 2
        target_01 = (target + 1) / 2
        
        # Calculate metrics
        mse = F.mse_loss(pred_01, target_01).item()
        mae = F.l1_loss(pred_01, target_01).item()
        psnr = calculate_psnr(pred_01, target_01, max_val=1.0)
        ssim = calculate_ssim(pred_01, target_01, max_val=1.0)
        
        return {
            'mse': mse,
            'mae': mae,
            'psnr': psnr,
            'ssim': ssim
        }


if __name__ == "__main__":
    # Test losses
    print("Testing loss functions...\n")
    
    # Test Pix2Pix loss
    print("Pix2Pix Loss:")
    pix2pix_loss = Pix2PixLoss(lambda_l1=100.0)
    fake_pred = torch.randn(2, 1, 15, 15)
    fake_img = torch.randn(2, 3, 256, 256)
    real_img = torch.randn(2, 3, 256, 256)
    real_pred = torch.randn(2, 1, 15, 15)
    
    g_loss, gan_loss, l1_loss = pix2pix_loss.generator_loss(fake_pred, fake_img, real_img)
    d_loss, real_loss, fake_loss = pix2pix_loss.discriminator_loss(real_pred, fake_pred)
    print(f"Generator loss: {g_loss:.4f} (GAN: {gan_loss:.4f}, L1: {l1_loss:.4f})")
    print(f"Discriminator loss: {d_loss:.4f} (Real: {real_loss:.4f}, Fake: {fake_loss:.4f})\n")
    
    # Test VAE loss
    print("VAE Loss:")
    vae_loss = VAELoss()
    recon = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    mu = torch.randn(2, 512)
    logvar = torch.randn(2, 512)
    
    total, recon_l, kld = vae_loss(recon, target, mu, logvar)
    print(f"Total loss: {total:.4f} (Recon: {recon_l:.4f}, KLD: {kld:.4f})\n")
    
    # Test metrics
    print("Metrics:")
    metrics_calc = MetricsCalculator()
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    metrics = metrics_calc.calculate(pred, target)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\nLoss functions test completed!")
