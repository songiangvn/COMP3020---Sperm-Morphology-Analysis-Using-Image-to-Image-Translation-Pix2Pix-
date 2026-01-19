"""
Enhanced Generator for Non-Real-Time Virtual Staining
Maximizes quality over speed for offline training/inference

Key Improvements over Paper's Lightweight Model:
1. Full standard convolutions instead of depthwise separable
2. Increased channel capacity (up to 1024 in bottleneck)
3. Attention mechanisms (CBAM) for better feature selection
4. Transposed convolutions for learnable upsampling
5. Residual connections for better gradient flow
6. Enhanced SPPF with more scales
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module (from CBAM)"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention Module (from CBAM)"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResidualBlock(nn.Module):
    """Residual Block with optional attention"""
    
    def __init__(self, channels: int, use_attention: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.attention = CBAM(channels) if use_attention else nn.Identity()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        out += residual
        out = self.relu(out)
        return out


class EnhancedVGGBlock(nn.Module):
    """Enhanced VGG Block with residual connection and attention"""
    
    def __init__(self, in_channels: int, out_channels: int, use_residual: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.use_residual = use_residual and (in_channels == out_channels)
        if use_residual and (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None
        
        self.attention = CBAM(out_channels)
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        
        if self.use_residual:
            if self.shortcut is not None:
                identity = self.shortcut(identity)
            out += identity
        
        out = self.relu(out)
        return out


class EnhancedSPPF(nn.Module):
    """Enhanced Spatial Pyramid Pooling Fusion
    
    Improvements:
    - More pooling scales (4 instead of 3)
    - Mixed pooling (max + avg)
    - Larger kernel sizes for better receptive field
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden_channels = in_channels // 2
        
        # Initial reduction
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        # Multi-scale pooling (4 scales)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.pool3 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool4 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)  # Add avg pooling
        
        # Concatenate: hidden + 4 pooled = 5 * hidden
        self.conv_out = nn.Conv2d(5 * hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        pool1 = self.pool1(x)
        pool2 = self.pool2(pool1)
        pool3 = self.pool3(pool2)
        pool4 = self.pool4(x)
        
        out = torch.cat([x, pool1, pool2, pool3, pool4], dim=1)
        out = self.relu(self.bn_out(self.conv_out(out)))
        
        return out


class EnhancedDecoderBlock(nn.Module):
    """Enhanced Decoder Block with standard convolutions and attention"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Two conv layers like paper
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.attention = CBAM(out_channels)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.attention(x)
        return x


class EnhancedUNetGenerator(nn.Module):
    """Enhanced U-Net Generator for Maximum Quality
    
    Key Differences from Paper's Lightweight Model:
    - Standard convolutions (no depthwise separable)
    - Larger channel capacity: 64, 128, 256, 512, 1024
    - CBAM attention at every level
    - Transposed conv for upsampling (learnable)
    - Residual connections in encoder
    
    Expected parameters: ~45-50M (vs 0.831M in paper)
    Expected quality: Much higher PSNR/SSIM
    Training time: ~5-10x slower (acceptable for offline use)
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64):
        super().__init__()
        
        # Channel progression
        c1 = base_channels      # 64
        c2 = c1 * 2             # 128
        c3 = c2 * 2             # 256
        c4 = c3 * 2             # 512
        c5 = c4 * 2             # 1024
        
        # Initial block
        self.init_block = EnhancedVGGBlock(in_channels, c1, use_residual=False)
        
        # Encoder with Enhanced SPPF and residual blocks
        self.enc1_res = ResidualBlock(c1)
        self.enc1 = EnhancedSPPF(c1, c2)
        self.down1 = nn.MaxPool2d(2)
        
        self.enc2_res = ResidualBlock(c2)
        self.enc2 = EnhancedSPPF(c2, c3)
        self.down2 = nn.MaxPool2d(2)
        
        self.enc3_res = ResidualBlock(c3)
        self.enc3 = EnhancedSPPF(c3, c4)
        self.down3 = nn.MaxPool2d(2)
        
        self.enc4_res = ResidualBlock(c4)
        self.enc4 = EnhancedSPPF(c4, c5)
        self.down4 = nn.MaxPool2d(2)
        
        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            EnhancedSPPF(c5, c5),
            ResidualBlock(c5),
            ResidualBlock(c5)
        )
        
        # Decoder with transposed convolutions
        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = EnhancedDecoderBlock(c4 + c5, c4)  # c4 from up + c5 from skip
        
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = EnhancedDecoderBlock(c3 + c4, c3)  # c3 from up + c4 from skip
        
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = EnhancedDecoderBlock(c2 + c3, c2)  # c2 from up + c3 from skip
        
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = EnhancedDecoderBlock(c1 + c2, c1)  # c1 from up + c2 from skip
        
        # Final output
        self.final = nn.Sequential(
            EnhancedDecoderBlock(c1 + c1, c1),  # With skip from init
            nn.Conv2d(c1, out_channels, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Initial
        init = self.init_block(x)  # 256x256 x c1
        
        # Encoder
        e1 = self.enc1_res(init)
        e1 = self.enc1(e1)         # 256x256 x c2
        d1 = self.down1(e1)        # 128x128 x c2
        
        e2 = self.enc2_res(d1)
        e2 = self.enc2(e2)         # 128x128 x c3
        d2 = self.down2(e2)        # 64x64 x c3
        
        e3 = self.enc3_res(d2)
        e3 = self.enc3(e3)         # 64x64 x c4
        d3 = self.down3(e3)        # 32x32 x c4
        
        e4 = self.enc4_res(d3)
        e4 = self.enc4(e4)         # 32x32 x c5
        d4 = self.down4(e4)        # 16x16 x c5
        
        # Bottleneck
        b = self.bottleneck(d4)    # 16x16 x c5
        
        # Decoder with skip connections
        u4 = self.up4(b)                     # 32x32 x c4
        u4 = torch.cat([u4, e4], dim=1)      # 32x32 x (c4+c4)
        u4 = self.dec4(u4)                   # 32x32 x c4
        
        u3 = self.up3(u4)                    # 64x64 x c3
        u3 = torch.cat([u3, e3], dim=1)      # 64x64 x (c3+c3)
        u3 = self.dec3(u3)                   # 64x64 x c3
        
        u2 = self.up2(u3)                    # 128x128 x c2
        u2 = torch.cat([u2, e2], dim=1)      # 128x128 x (c2+c2)
        u2 = self.dec2(u2)                   # 128x128 x c2
        
        u1 = self.up1(u2)                    # 256x256 x c1
        u1 = torch.cat([u1, e1], dim=1)      # 256x256 x (c1+c1)
        u1 = self.dec1(u1)                   # 256x256 x c1
        
        # Final with skip from init
        out = torch.cat([u1, init], dim=1)   # 256x256 x (c1+c1)
        out = self.final(out)                 # 256x256 x out_channels
        
        return out


class EnhancedDiscriminator(nn.Module):
    """Enhanced PatchGAN Discriminator with spectral normalization"""
    
    def __init__(self, in_channels: int = 6, base_filters: int = 64):
        super().__init__()
        
        def discriminator_block(in_f, out_f, stride=2, normalize=True, use_spectral_norm=True):
            layers = []
            conv = nn.Conv2d(in_f, out_f, kernel_size=4, stride=stride, padding=1)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers.append(conv)
            
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            discriminator_block(in_channels, base_filters, normalize=False),  # 128x128
            discriminator_block(base_filters, base_filters * 2),               # 64x64
            discriminator_block(base_filters * 2, base_filters * 4),           # 32x32
            discriminator_block(base_filters * 4, base_filters * 8, stride=1), # 32x32
            nn.Conv2d(base_filters * 8, 1, kernel_size=4, padding=1),         # 31x31
            nn.Sigmoid()
        )
    
    def forward(self, img_input, img_target):
        x = torch.cat([img_input, img_target], dim=1)
        return self.model(x)


def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Test model
    print("Testing Enhanced Generator...")
    model = EnhancedUNetGenerator(in_channels=3, out_channels=3, base_channels=64)
    
    # Count parameters
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Test discriminator
    print("\nTesting Enhanced Discriminator...")
    disc = EnhancedDiscriminator(in_channels=6, base_filters=64)
    total_d, trainable_d = count_parameters(disc)
    print(f"Discriminator parameters: {total_d:,}")
    
    with torch.no_grad():
        pred = disc(x, y)
    print(f"Discriminator output shape: {pred.shape}")
    print(f"Discriminator output range: [{pred.min():.3f}, {pred.max():.3f}]")
