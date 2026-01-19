"""
Baseline models for comparison:
1. Standard cGAN (conditional GAN)
2. VAE (Variational Autoencoder)
3. U-Net (without GAN)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============= Standard cGAN (Conditional GAN) =============

class StandardUNetGenerator(nn.Module):
    """Standard U-Net Generator for cGAN"""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        
        def down_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)
        
        def up_block(in_feat, out_feat, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)
        
        # Encoder
        self.down1 = down_block(in_channels, 64, normalize=False)  # 128x128
        self.down2 = down_block(64, 128)                            # 64x64
        self.down3 = down_block(128, 256)                           # 32x32
        self.down4 = down_block(256, 512)                           # 16x16
        self.down5 = down_block(512, 512)                           # 8x8
        self.down6 = down_block(512, 512)                           # 4x4
        self.down7 = down_block(512, 512)                           # 2x2
        self.down8 = down_block(512, 512, normalize=False)          # 1x1
        
        # Decoder
        self.up1 = up_block(512, 512, dropout=True)                # 2x2
        self.up2 = up_block(1024, 512, dropout=True)               # 4x4
        self.up3 = up_block(1024, 512, dropout=True)               # 8x8
        self.up4 = up_block(1024, 512)                             # 16x16
        self.up5 = up_block(1024, 256)                             # 32x32
        self.up6 = up_block(512, 128)                              # 64x64
        self.up7 = up_block(256, 64)                               # 128x128
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder with skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # Decoder with skip connections
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        
        return self.final(torch.cat([u7, d1], 1))


class StandardDiscriminator(nn.Module):
    """Standard PatchGAN Discriminator"""
    
    def __init__(self, in_channels: int = 6):
        super().__init__()
        
        def discriminator_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            discriminator_block(in_channels, 64, normalize=False),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, img_input, img_target):
        img_concat = torch.cat([img_input, img_target], 1)
        return self.model(img_concat)


class StandardcGAN(nn.Module):
    """Standard conditional GAN"""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.generator = StandardUNetGenerator(in_channels, out_channels)
        self.discriminator = StandardDiscriminator(in_channels + out_channels)
    
    def forward(self, x):
        return self.generator(x)


# ============= VAE (Variational Autoencoder) =============

class VAEEncoder(nn.Module):
    """VAE Encoder"""
    
    def __init__(self, in_channels: int = 3, latent_dim: int = 512):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Latent space: mu and log_var
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder"""
    
    def __init__(self, latent_dim: int = 512, out_channels: int = 3):
        super().__init__()
        
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 8, 8)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    """Variational Autoencoder for image-to-image translation"""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, latent_dim: int = 512):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(latent_dim, out_channels)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


# ============= Standard U-Net (without GAN) =============

class SimpleUNet(nn.Module):
    """Simple U-Net without adversarial training"""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        
        def double_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        def down(in_ch, out_ch):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch)
            )
        
        def up(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        self.inc = double_conv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        
        self.up1 = up(512, 512)
        self.conv1 = double_conv(1024, 512)
        
        self.up2 = up(512, 256)
        self.conv2 = double_conv(512, 256)
        
        self.up3 = up(256, 128)
        self.conv3 = double_conv(256, 128)
        
        self.up4 = up(128, 64)
        self.conv4 = double_conv(128, 64)
        
        self.outc = nn.Sequential(
            nn.Conv2d(64, out_channels, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        
        x = self.outc(x)
        return x


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256).to(device)
    
    print("Testing baseline models...\n")
    
    # Test cGAN
    print("=" * 50)
    print("1. Standard cGAN")
    print("=" * 50)
    cgan = StandardcGAN().to(device)
    output = cgan(x)
    print(f"Generator parameters: {count_parameters(cgan.generator):,}")
    print(f"Discriminator parameters: {count_parameters(cgan.discriminator):,}")
    print(f"Output shape: {output.shape}\n")
    
    # Test VAE
    print("=" * 50)
    print("2. VAE")
    print("=" * 50)
    vae = VAE().to(device)
    output, mu, logvar = vae(x)
    print(f"Parameters: {count_parameters(vae):,}")
    print(f"Output shape: {output.shape}")
    print(f"Latent space: mu={mu.shape}, logvar={logvar.shape}\n")
    
    # Test Simple U-Net
    print("=" * 50)
    print("3. Simple U-Net")
    print("=" * 50)
    unet = SimpleUNet().to(device)
    output = unet(x)
    print(f"Parameters: {count_parameters(unet):,}")
    print(f"Output shape: {output.shape}\n")
    
    print("All baseline models tested successfully!")
