"""
Model factory and initialization
"""
from models.pix2pix_modified import Pix2PixModified, ModifiedUNetGenerator, PatchGANDiscriminator
from models.baselines import StandardcGAN, VAE, SimpleUNet
from models.enhanced_generator import EnhancedUNetGenerator, EnhancedDiscriminator


def get_model(model_name: str, **kwargs):
    """
    Factory function to get model by name
    
    Args:
        model_name: One of ['pix2pix_modified', 'pix2pix_enhanced', 'cgan', 'vae', 'unet']
        **kwargs: Additional arguments for model initialization
    
    Returns:
        model instance
    """
    model_dict = {
        'pix2pix_modified': Pix2PixModified,
        'pix2pix_enhanced': None,  # Use get_generator/get_discriminator
        'cgan': StandardcGAN,
        'vae': VAE,
        'unet': SimpleUNet,
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_dict.keys())}")
    
    return model_dict[model_name](**kwargs)


def get_generator(model_name: str, **kwargs):
    """Get only the generator part"""
    if model_name == 'pix2pix_modified':
        return ModifiedUNetGenerator(**kwargs)
    elif model_name == 'pix2pix_enhanced':
        return EnhancedUNetGenerator(**kwargs)
    elif model_name == 'cgan':
        from models.baselines import StandardUNetGenerator
        return StandardUNetGenerator(**kwargs)
    elif model_name == 'vae':
        return VAE(**kwargs)
    elif model_name == 'unet':
        return SimpleUNet(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_discriminator(model_name: str, **kwargs):
    """Get only the discriminator part (if applicable)"""
    if model_name == 'pix2pix_modified':
        return PatchGANDiscriminator(**kwargs)
    elif model_name == 'pix2pix_enhanced':
        return EnhancedDiscriminator(**kwargs)
    elif model_name == 'cgan':
        from models.baselines import StandardDiscriminator
        return StandardDiscriminator(**kwargs)
    elif model_name in ['vae', 'unet']:
        return None  # These don't use discriminator
    else:
        raise ValueError(f"Unknown model: {model_name}")
