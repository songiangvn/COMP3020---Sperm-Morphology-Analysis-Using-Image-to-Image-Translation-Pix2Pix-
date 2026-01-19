"""
Baseline models for virtual staining comparison
"""
from .unet import UNet
from .cgan import Generator as cGAN_Generator, Discriminator as cGAN_Discriminator
from .vae import VAE, vae_loss

__all__ = [
    'UNet',
    'cGAN_Generator',
    'cGAN_Discriminator',
    'VAE',
    'vae_loss'
]
