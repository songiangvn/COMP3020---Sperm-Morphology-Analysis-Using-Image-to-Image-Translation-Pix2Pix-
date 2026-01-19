"""
U-Net baseline (without GAN)
Simple encoder-decoder for direct image translation
"""
import torch
import torch.nn as nn


class UNet(nn.Module):
    """Standard U-Net for image-to-image translation"""
    
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        def down_block(in_feat, out_feat):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, 3, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.3),  # Added dropout to prevent overfitting
                nn.Conv2d(out_feat, out_feat, 3, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.3)   # Added dropout
            )
        
        def up_block(in_feat, out_feat):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, 3, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),  # Lighter dropout in decoder
                nn.Conv2d(out_feat, out_feat, 3, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True)
            )
        
        # Encoder
        self.enc1 = down_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = down_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = down_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = down_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = down_block(512, 1024)
        
        # Decoder - Use Upsample+Conv to avoid checkerboard artifacts
        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(1024, 512, 3, padding=1)
        )
        self.dec4 = up_block(1024, 512)
        
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, padding=1)
        )
        self.dec3 = up_block(512, 256)
        
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, padding=1)
        )
        self.dec2 = up_block(256, 128)
        
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1)
        )
        self.dec1 = up_block(128, 64)
        
        # Final output
        self.out = nn.Conv2d(64, out_channels, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.tanh(self.out(d1))
