"""
Train 3 baseline models: UNet, cGAN, VAE
For comparison with Pix2Pix (HoloStain)
"""
import os
import sys
import time
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Import baseline models
from baselines.unet import UNet
from baselines.cgan import Generator as cGAN_Generator, Discriminator as cGAN_Discriminator
from baselines.vae import VAE, vae_loss


class PairedImageDataset(Dataset):
    """Dataset for paired holography and stained images"""
    
    def __init__(self, data_dir, image_size=256, mode='train'):
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Get all PHASE files (input)
        phase_files = sorted([f for f in os.listdir(data_dir) if 'PHASE' in f and f.endswith('.png')])
        
        # Split train/val/test = 70/15/15
        n = len(phase_files)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        if mode == 'train':
            self.phase_files = phase_files[:train_end]
        elif mode == 'val':
            self.phase_files = phase_files[train_end:val_end]
        else:  # test
            self.phase_files = phase_files[val_end:]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.phase_files)
    
    def __getitem__(self, idx):
        # Load PHASE (input)
        phase_file = self.phase_files[idx]
        phase_path = os.path.join(self.data_dir, phase_file)
        holography = Image.open(phase_path).convert('RGB')
        
        # Load corresponding STAIN (ground truth)
        stain_file = phase_file.replace('PHASE', 'STAIN')
        stain_path = os.path.join(self.data_dir, stain_file)
        stained = Image.open(stain_path).convert('RGB')
        
        holography = self.transform(holography)
        stained = self.transform(stained)
        
        return holography, stained


def train_unet(args):
    """Train UNet baseline"""
    print("\n" + "="*50)
    print("Training UNet Baseline")
    print("="*50)
    
    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.join(args.output_dir, 'unet')
    os.makedirs(output_dir, exist_ok=True)
    
    # Data
    train_dataset = PairedImageDataset(args.data_dir, args.image_size, mode='train')
    val_dataset = PairedImageDataset(args.data_dir, args.image_size, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model
    model = UNet(in_channels=3, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    
    # Training loop
    best_val_loss = float('inf')
    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for i, (holo, stained) in enumerate(train_loader):
            holo, stained = holo.to(device), stained.to(device)
            
            # Forward pass
            output = model(holo)
            loss = criterion(output, stained)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for holography, stained in val_loader:
                holography, stained = holography.to(device), stained.to(device)
                output = model(holography)
                loss = criterion(output, stained)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(output_dir, 'unet_best.pth'))
            print(f"✓ Saved new best model! Val Loss: {avg_val_loss:.4f}")
    
    print(f"UNet training completed! Best model saved to {output_dir}/unet_best.pth (Val Loss: {best_val_loss:.4f})")


def train_cgan(args):
    """Train conditional GAN baseline"""
    print("\n" + "="*50)
    print("Training cGAN Baseline")
    print("="*50)
    
    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.join(args.output_dir, 'cgan')
    os.makedirs(output_dir, exist_ok=True)
    
    # Data
    train_dataset = PairedImageDataset(args.data_dir, args.image_size, mode='train')
    val_dataset = PairedImageDataset(args.data_dir, args.image_size, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Models
    generator = cGAN_Generator(in_channels=3, out_channels=3).to(device)
    discriminator = cGAN_Discriminator(in_channels=6).to(device)
    
    # Optimizers - Reduce D learning rate to balance with G
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr * 0.1, betas=(0.5, 0.999))  # 10x slower
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    lambda_L1 = 100  # Weight for L1 loss
    
    # Training loop
    best_val_loss = float('inf')
    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        for i, (holo, stained) in enumerate(train_loader):
            holo, stained = holo.to(device), stained.to(device)
            batch_size = holo.size(0)
            
            # ==================
            # Train Discriminator
            # ==================
            optimizer_D.zero_grad()
            
            # Real images - Add label smoothing to stabilize training
            d_real = discriminator(holo, stained)
            
            # Label smoothing: 0.9 instead of 1.0 for real, 0.1 instead of 0.0 for fake
            real_label = torch.ones_like(d_real).to(device) * 0.9
            fake_label = torch.zeros_like(d_real).to(device) + 0.1
            
            loss_d_real = criterion_GAN(d_real, real_label)
            
            # Fake images
            fake_stained = generator(holo)
            d_fake = discriminator(holo, fake_stained.detach())
            loss_d_fake = criterion_GAN(d_fake, fake_label)
            
            # Total discriminator loss
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_D.step()
            
            # ==================
            # Train Generator
            # ==================
            optimizer_G.zero_grad()
            
            # GAN loss - use hard labels (1.0) for generator
            d_fake = discriminator(holo, fake_stained)
            real_label_G = torch.ones_like(d_fake).to(device)  # Generator wants D to output 1.0
            loss_g_gan = criterion_GAN(d_fake, real_label_G)
            
            # L1 loss
            loss_g_l1 = criterion_L1(fake_stained, stained)
            
            # Total generator loss
            loss_g = loss_g_gan + lambda_L1 * loss_g_l1
            loss_g.backward()
            optimizer_G.step()
            
            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            
            if (i + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{i+1}/{len(train_loader)}] "
                      f"D_loss: {loss_d.item():.4f} G_loss: {loss_g.item():.4f}")
        
        avg_train_g_loss = epoch_g_loss / len(train_loader)
        avg_train_d_loss = epoch_d_loss / len(train_loader)
        
        # Validation
        generator.eval()
        discriminator.eval()
        val_g_loss = 0
        val_d_loss = 0
        
        with torch.no_grad():
            for holo, stained in val_loader:
                holo, stained = holo.to(device), stained.to(device)
                
                # Generate fake images
                fake_stained = generator(holo)
                
                # Discriminator evaluation
                d_real = discriminator(holo, stained)
                d_fake = discriminator(holo, fake_stained)
                
                real_label = torch.ones_like(d_real).to(device)
                fake_label = torch.zeros_like(d_fake).to(device)
                
                loss_d_real = criterion_GAN(d_real, real_label)
                loss_d_fake = criterion_GAN(d_fake, fake_label)
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                
                # Generator evaluation
                loss_g_gan = criterion_GAN(d_fake, real_label)
                loss_g_l1 = criterion_L1(fake_stained, stained)
                loss_g = loss_g_gan + lambda_L1 * loss_g_l1
                
                val_g_loss += loss_g.item()
                val_d_loss += loss_d.item()
        
        avg_val_g_loss = val_g_loss / len(val_loader)
        avg_val_d_loss = val_d_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] Train G: {avg_train_g_loss:.4f} D: {avg_train_d_loss:.4f} | "
              f"Val G: {avg_val_g_loss:.4f} D: {avg_val_d_loss:.4f}")
        
        # Save best model based on validation G loss
        if avg_val_g_loss < best_val_loss:
            best_val_loss = avg_val_g_loss
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'train_g_loss': avg_train_g_loss,
                'train_d_loss': avg_train_d_loss,
                'val_g_loss': avg_val_g_loss,
                'val_d_loss': avg_val_d_loss,
            }, os.path.join(output_dir, 'cgan_best.pth'))
            print(f"✓ Saved new best model! Val G Loss: {avg_val_g_loss:.4f}")
    
    print(f"cGAN training completed! Best model saved to {output_dir}/cgan_best.pth (Val G Loss: {best_val_loss:.4f})")


def train_vae(args):
    """Train VAE baseline"""
    print("\n" + "="*50)
    print("Training VAE Baseline")
    print("="*50)
    
    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.join(args.output_dir, 'vae')
    os.makedirs(output_dir, exist_ok=True)
    
    # Data
    train_dataset = PairedImageDataset(args.data_dir, args.image_size, mode='train')
    val_dataset = PairedImageDataset(args.data_dir, args.image_size, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model
    model = VAE(in_channels=3, out_channels=3, latent_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kld_loss = 0
        
        for i, (holo, stained) in enumerate(train_loader):
            holo, stained = holo.to(device), stained.to(device)
            
            # Forward pass
            recon, mu, logvar = model(holo)
            
            # Loss
            loss, recon_loss, kld_loss = vae_loss(recon, stained, mu, logvar)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kld_loss += kld_loss.item()
            
            if (i + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{i+1}/{len(train_loader)}] "
                      f"Total: {loss.item():.4f} Recon: {recon_loss.item():.4f} KLD: {kld_loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_recon = epoch_recon_loss / len(train_loader)
        avg_train_kld = epoch_kld_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kld_loss = 0
        
        with torch.no_grad():
            for holo, stained in val_loader:
                holo, stained = holo.to(device), stained.to(device)
                recon, mu, logvar = model(holo)
                loss, recon_loss, kld_loss = vae_loss(recon, stained, mu, logvar)
                
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kld_loss += kld_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon = val_recon_loss / len(val_loader)
        avg_val_kld = val_kld_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] Train Total: {avg_train_loss:.4f} Recon: {avg_train_recon:.4f} | "
              f"Val Total: {avg_val_loss:.4f} Recon: {avg_val_recon:.4f}")
        
        # Save best model based on validation total loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'train_recon': avg_train_recon,
                'train_kld': avg_train_kld,
                'val_loss': avg_val_loss,
                'val_recon': avg_val_recon,
                'val_kld': avg_val_kld,
            }, os.path.join(output_dir, 'vae_best.pth'))
            print(f"✓ Saved new best model! Val Loss: {avg_val_loss:.4f}")
    
    print(f"VAE training completed! Best model saved to {output_dir}/vae_best.pth (Val Loss: {best_val_loss:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--model', type=str, required=True, choices=['unet', 'cgan', 'vae', 'all'],
                        help='Model to train (unet/cgan/vae/all)')
    parser.add_argument('--data_dir', type=str, default='/home/23giang.ns/ML_Project/virtual_stainning/both',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs_baselines',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=120,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    args = parser.parse_args()
    
    # Train specified model(s)
    if args.model == 'unet' or args.model == 'all':
        train_unet(args)
    
    if args.model == 'cgan' or args.model == 'all':
        train_cgan(args)
    
    if args.model == 'vae' or args.model == 'all':
        train_vae(args)
    
    print("\n" + "="*50)
    print("All training completed!")
    print("="*50)


if __name__ == '__main__':
    main()
