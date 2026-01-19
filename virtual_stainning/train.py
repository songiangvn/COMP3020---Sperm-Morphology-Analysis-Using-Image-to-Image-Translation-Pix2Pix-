"""
Training script for virtual staining models
"""
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from pathlib import Path
import numpy as np

from dataset import get_dataloaders
from models import get_generator, get_discriminator
from losses import get_loss_function, MetricsCalculator


class Trainer:
    """Base trainer class"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        self.output_dir = Path(config['output_dir']) / config['model_name'] / config['exp_name']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        # Initialize models
        self.setup_models()
        
        # Initialize optimizers
        self.setup_optimizers()
        
        # Initialize loss functions
        self.setup_losses()
        
        # Get dataloaders
        self.train_loader, self.val_loader = get_dataloaders(
            root_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            val_ratio=config['val_ratio']
        )
        
        # Metrics
        self.metrics_calc = MetricsCalculator()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_psnr = 0.0  # Track best PSNR (higher is better)
    
    def setup_models(self):
        """Setup models - to be implemented by subclasses"""
        raise NotImplementedError
    
    def setup_optimizers(self):
        """Setup optimizers - to be implemented by subclasses"""
        raise NotImplementedError
    
    def setup_losses(self):
        """Setup loss functions - to be implemented by subclasses"""
        raise NotImplementedError
    
    def train_epoch(self):
        """Train one epoch - to be implemented by subclasses"""
        raise NotImplementedError
    
    def validate(self):
        """Validate - to be implemented by subclasses"""
        raise NotImplementedError
    
    def save_checkpoint(self, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        
        # Add model states
        if hasattr(self, 'generator'):
            checkpoint['generator'] = self.generator.state_dict()
        if hasattr(self, 'discriminator') and self.discriminator is not None:
            checkpoint['discriminator'] = self.discriminator.state_dict()
        if hasattr(self, 'model'):
            checkpoint['model'] = self.model.state_dict()
        
        # Add optimizer states
        if hasattr(self, 'optimizer_g'):
            checkpoint['optimizer_g'] = self.optimizer_g.state_dict()
        if hasattr(self, 'optimizer_d') and self.optimizer_d is not None:
            checkpoint['optimizer_d'] = self.optimizer_d.state_dict()
        if hasattr(self, 'optimizer'):
            checkpoint['optimizer'] = self.optimizer.state_dict()
        
        # Save
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save as latest
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save as best
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"✅ Saved BEST model: PSNR={self.best_psnr:.2f} dB, val_loss={self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_psnr = checkpoint.get('best_psnr', 0.0)  # Backward compatible
        
        if hasattr(self, 'generator') and 'generator' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator'])
        if hasattr(self, 'discriminator') and self.discriminator is not None and 'discriminator' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator'])
        if hasattr(self, 'model') and 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        
        if hasattr(self, 'optimizer_g') and 'optimizer_g' in checkpoint:
            self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        if hasattr(self, 'optimizer_d') and self.optimizer_d is not None and 'optimizer_d' in checkpoint:
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        if hasattr(self, 'optimizer') and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on device: {self.device}")
        print(f"Training for {self.config['num_epochs']} epochs")
        
        for epoch in range(self.epoch, self.config['num_epochs']):
            self.epoch = epoch
            
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train
            train_losses = self.train_epoch()
            
            # Log training losses
            for k, v in train_losses.items():
                self.writer.add_scalar(f'train/{k}', v, epoch)
            
            # Validate
            if (epoch + 1) % self.config['val_freq'] == 0:
                val_losses, val_metrics = self.validate()
                
                # Log validation losses and metrics
                for k, v in val_losses.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)
                
                # Check if best model based on PSNR (higher is better for image quality)
                current_val_loss = val_losses.get('total_loss', val_losses.get('loss', float('inf')))
                current_psnr = val_metrics['psnr']
                is_best = current_psnr > self.best_psnr
                if is_best:
                    self.best_val_loss = current_val_loss
                    self.best_psnr = current_psnr
                
                # Print validation results
                print(f"Val Loss: {current_val_loss:.4f}")
                print(f"Val Metrics: PSNR={val_metrics['psnr']:.2f}, SSIM={val_metrics['ssim']:.4f}")
                if is_best:
                    print(f"🔥 New best PSNR: {self.best_psnr:.2f} dB")
            else:
                is_best = False
            
            # Save checkpoint: always save if best, or according to save_freq
            if is_best or (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(is_best=is_best)
        
        print("\nTraining completed!")
        self.writer.close()


class GANTrainer(Trainer):
    """Trainer for GAN-based models (Pix2Pix, cGAN)"""
    
    def setup_models(self):
        self.generator = get_generator(
            self.config['model_name'],
            in_channels=3,
            out_channels=3
        ).to(self.device)
        
        self.discriminator = get_discriminator(
            self.config['model_name'],
            in_channels=6
        ).to(self.device)
        
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def setup_optimizers(self):
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.config['lr_g'],
            betas=(self.config['beta1'], 0.999)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config['lr_d'],
            betas=(self.config['beta1'], 0.999)
        )
    
    def setup_losses(self):
        from losses import Pix2PixLoss
        self.criterion = Pix2PixLoss(
            lambda_l1=self.config['lambda_l1'],
            use_lsgan=self.config['use_lsgan']
        )
        
        # Move perceptual loss to device if it exists
        if hasattr(self.criterion, 'perceptual_loss') and self.criterion.perceptual_loss is not None:
            self.criterion.perceptual_loss = self.criterion.perceptual_loss.to(self.device)
    
    def train_epoch(self):
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        total_gan_loss = 0
        total_l1_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            phase = batch['phase'].to(self.device)
            stain = batch['stain'].to(self.device)
            
            # Train Discriminator
            self.optimizer_d.zero_grad()
            
            fake_stain = self.generator(phase)
            
            real_pred = self.discriminator(phase, stain)
            fake_pred = self.discriminator(phase, fake_stain.detach())
            
            d_loss, real_loss, fake_loss = self.criterion.discriminator_loss(real_pred, fake_pred)
            d_loss.backward()
            self.optimizer_d.step()
            
            # Train Generator
            self.optimizer_g.zero_grad()
            
            fake_pred = self.discriminator(phase, fake_stain)
            g_loss, gan_loss, l1_loss, perceptual_loss = self.criterion.generator_loss(fake_pred, fake_stain, stain)
            g_loss.backward()
            self.optimizer_g.step()
            
            # Accumulate losses
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            total_gan_loss += gan_loss.item()
            total_l1_loss += l1_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G': f'{g_loss.item():.4f}',
                'D': f'{d_loss.item():.4f}',
                'L1': f'{l1_loss.item():.4f}'
            })
        
        n = len(self.train_loader)
        return {
            'g_loss': total_g_loss / n,
            'd_loss': total_d_loss / n,
            'gan_loss': total_gan_loss / n,
            'l1_loss': total_l1_loss / n
        }
    
    @torch.no_grad()
    def validate(self):
        self.generator.eval()
        
        total_g_loss = 0
        all_metrics = {'mse': 0, 'mae': 0, 'psnr': 0, 'ssim': 0}
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            phase = batch['phase'].to(self.device)
            stain = batch['stain'].to(self.device)
            
            fake_stain = self.generator(phase)
            
            # Calculate metrics
            metrics = self.metrics_calc.calculate(fake_stain, stain)
            for k, v in metrics.items():
                all_metrics[k] += v
            
            # Calculate loss
            fake_pred = self.discriminator(phase, fake_stain)
            g_loss, _, _, _ = self.criterion.generator_loss(fake_pred, fake_stain, stain)
            total_g_loss += g_loss.item()
        
        n = len(self.val_loader)
        losses = {'total_loss': total_g_loss / n}
        metrics = {k: v / n for k, v in all_metrics.items()}
        
        return losses, metrics


class VAETrainer(Trainer):
    """Trainer for VAE"""
    
    def setup_models(self):
        from models.baselines import VAE
        self.model = VAE(in_channels=3, out_channels=3).to(self.device)
        print(f"VAE parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_optimizers(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['lr_g'],
            betas=(self.config['beta1'], 0.999)
        )
    
    def setup_losses(self):
        from losses import VAELoss
        self.criterion = VAELoss(kld_weight=self.config.get('kld_weight', 0.00025))
    
    def train_epoch(self):
        self.model.train()
        
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            phase = batch['phase'].to(self.device)
            stain = batch['stain'].to(self.device)
            
            self.optimizer.zero_grad()
            
            recon, mu, logvar = self.model(phase)
            loss, recon_loss, kld_loss = self.criterion(recon, stain, mu, logvar)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'KLD': f'{kld_loss.item():.4f}'
            })
        
        n = len(self.train_loader)
        return {
            'loss': total_loss / n,
            'recon_loss': total_recon / n,
            'kld_loss': total_kld / n
        }
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        
        total_loss = 0
        all_metrics = {'mse': 0, 'mae': 0, 'psnr': 0, 'ssim': 0}
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            phase = batch['phase'].to(self.device)
            stain = batch['stain'].to(self.device)
            
            recon, mu, logvar = self.model(phase)
            loss, _, _ = self.criterion(recon, stain, mu, logvar)
            
            total_loss += loss.item()
            
            metrics = self.metrics_calc.calculate(recon, stain)
            for k, v in metrics.items():
                all_metrics[k] += v
        
        n = len(self.val_loader)
        losses = {'loss': total_loss / n}
        metrics = {k: v / n for k, v in all_metrics.items()}
        
        return losses, metrics


class UNetTrainer(Trainer):
    """Trainer for simple U-Net"""
    
    def setup_models(self):
        from models.baselines import SimpleUNet
        self.model = SimpleUNet(in_channels=3, out_channels=3).to(self.device)
        print(f"U-Net parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_optimizers(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['lr_g'],
            betas=(self.config['beta1'], 0.999)
        )
    
    def setup_losses(self):
        from losses import SimpleLoss
        self.criterion = SimpleLoss()
    
    def train_epoch(self):
        self.model.train()
        
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            phase = batch['phase'].to(self.device)
            stain = batch['stain'].to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(phase)
            loss = self.criterion(output, stain)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        n = len(self.train_loader)
        return {'loss': total_loss / n}
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        
        total_loss = 0
        all_metrics = {'mse': 0, 'mae': 0, 'psnr': 0, 'ssim': 0}
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            phase = batch['phase'].to(self.device)
            stain = batch['stain'].to(self.device)
            
            output = self.model(phase)
            loss = self.criterion(output, stain)
            
            total_loss += loss.item()
            
            metrics = self.metrics_calc.calculate(output, stain)
            for k, v in metrics.items():
                all_metrics[k] += v
        
        n = len(self.val_loader)
        losses = {'loss': total_loss / n}
        metrics = {k: v / n for k, v in all_metrics.items()}
        
        return losses, metrics


def get_trainer(config):
    """Get appropriate trainer based on model name"""
    if config['model_name'] in ['pix2pix_modified', 'pix2pix_enhanced', 'cgan']:
        return GANTrainer(config)
    elif config['model_name'] == 'vae':
        return VAETrainer(config)
    elif config['model_name'] == 'unet':
        return UNetTrainer(config)
    else:
        raise ValueError(f"Unknown model: {config['model_name']}")


def main():
    parser = argparse.ArgumentParser(description='Train virtual staining model')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['pix2pix_modified', 'pix2pix_enhanced', 'cgan', 'vae', 'unet'],
                       help='Model architecture')
    parser.add_argument('--exp_name', type=str, default='exp1', help='Experiment name')
    parser.add_argument('--data_dir', type=str, default='both', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr_g', type=float, default=0.0002, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.0002, help='Discriminator learning rate')
    parser.add_argument('--lambda_l1', type=float, default=100.0, help='L1 loss weight')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'model_name': args.model,
        'exp_name': args.exp_name,
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'lr_g': args.lr_g,
        'lr_d': args.lr_d,
        'lambda_l1': args.lambda_l1,
        'beta1': 0.5,
        'use_lsgan': True,
        'val_ratio': 0.15,
        'val_freq': 5,
        'save_freq': 10,
        'num_workers': args.num_workers,
        'kld_weight': 0.00025,
    }
    
    # Create trainer
    trainer = get_trainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
