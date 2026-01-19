"""
Preprocessing utilities for virtual staining
Handles the difference between 'both' (phase images) and 'unstained' datasets
"""
import cv2
import numpy as np
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageNormalizer:
    """Normalize images to have similar distribution"""
    
    @staticmethod
    def normalize_to_range(image: np.ndarray, target_min: float = 0.0, target_max: float = 1.0) -> np.ndarray:
        """Normalize image to target range"""
        img_min, img_max = image.min(), image.max()
        if img_max - img_min == 0:
            return np.zeros_like(image, dtype=np.float32)
        normalized = (image - img_min) / (img_max - img_min)
        return normalized * (target_max - target_min) + target_min
    
    @staticmethod
    def match_histogram(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match histogram of source to reference"""
        # Convert to grayscale if needed
        if len(source.shape) == 3:
            source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            source_gray = source
            reference_gray = reference
        
        # Match histogram
        matched = np.zeros_like(source)
        if len(source.shape) == 3:
            for i in range(3):
                matched[:, :, i] = ImageNormalizer._match_channel(source[:, :, i], reference_gray)
        else:
            matched = ImageNormalizer._match_channel(source, reference_gray)
        
        return matched
    
    @staticmethod
    def _match_channel(source_channel: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match single channel histogram"""
        # Calculate histograms
        s_hist, s_bins = np.histogram(source_channel.flatten(), 256, [0, 256])
        r_hist, r_bins = np.histogram(reference.flatten(), 256, [0, 256])
        
        # Calculate CDFs
        s_cdf = s_hist.cumsum()
        s_cdf = (255 * s_cdf / s_cdf[-1]).astype(np.uint8)
        
        r_cdf = r_hist.cumsum()
        r_cdf = (255 * r_cdf / r_cdf[-1]).astype(np.uint8)
        
        # Create lookup table
        lookup = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            j = np.argmin(np.abs(r_cdf - s_cdf[i]))
            lookup[i] = j
        
        return lookup[source_channel]
    
    @staticmethod
    def adjust_brightness_contrast(image: np.ndarray, 
                                   target_mean: float = 16.14, 
                                   target_std: float = 36.08) -> np.ndarray:
        """Adjust image to match target mean and std (matching 'both' dataset statistics)"""
        img_float = image.astype(np.float32)
        
        # Convert to grayscale if needed for statistics
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = img_float
        
        current_mean = gray.mean()
        current_std = gray.std()
        
        # Avoid division by zero
        if current_std == 0:
            return image
        
        # Standardize then scale to target distribution
        if len(image.shape) == 3:
            for i in range(3):
                channel = img_float[:, :, i]
                channel = (channel - current_mean) / current_std
                channel = channel * target_std + target_mean
                img_float[:, :, i] = np.clip(channel, 0, 255)
        else:
            img_float = (img_float - current_mean) / current_std
            img_float = img_float * target_std + target_mean
            img_float = np.clip(img_float, 0, 255)
        
        return img_float.astype(np.uint8)


class PreprocessingPipeline:
    """Complete preprocessing pipeline for virtual staining"""
    
    def __init__(self, mode: str = 'both', target_size: Tuple[int, int] = (256, 256)):
        """
        Args:
            mode: 'both' for training data or 'unstained' for inference data
            target_size: Target image size (H, W)
        """
        self.mode = mode
        self.target_size = target_size
        self.normalizer = ImageNormalizer()
        
        # Statistics from 'both' dataset
        self.both_mean = 16.14
        self.both_std = 36.08
    
    def preprocess_phase_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess phase (unstained) image
        
        Handles:
        - 8-bit RGB (training data from 'both')
        - 16-bit grayscale (inference data from 'unstained')
        - Different image sizes (256x256 vs 800x800)
        """
        # Handle 16-bit grayscale images (from unstained dataset)
        if image.dtype == np.uint16:
            # Normalize 16-bit to 8-bit range
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if self.mode == 'both':
            # For training data from 'both' dataset - just resize
            if image.shape[:2] != self.target_size:
                image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            return image
        else:
            # For unstained dataset - need to match distribution to 'both'
            # 1. Resize
            if image.shape[:2] != self.target_size:
                image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            
            # 2. Adjust brightness/contrast to match 'both' statistics
            image = self.normalizer.adjust_brightness_contrast(
                image, 
                target_mean=self.both_mean,
                target_std=self.both_std
            )
            
            return image
    
    def preprocess_stain_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess stain (target) image"""
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        return image


def get_train_augmentation(image_size: int = 256) -> A.Compose:
    """Get training augmentation pipeline - robust augmentation for better generalization"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),  # Reduced from 180, less aggressive
        
        # REDUCED: Lighter augmentation for sharper results
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),  # Was 0.3/0.7
        A.RandomGamma(gamma_limit=(90, 110), p=0.3),  # Was (80,120)/0.5
        
        # REDUCED: Less blur for better PSNR
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),  # Reduced blur limit
        ], p=0.2),  # Was 0.3
        
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),  # Reduced noise and probability
        
        # Color/intensity shifts - kept light
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3),  # Reduced
        
        # Normalize to [0, 1]
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ], additional_targets={'target': 'image'})


def get_val_augmentation(image_size: int = 256) -> A.Compose:
    """Get validation augmentation pipeline - only normalization"""
    return A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ], additional_targets={'target': 'image'})


def get_inference_augmentation(image_size: int = 256) -> A.Compose:
    """Get inference augmentation pipeline"""
    return A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])


if __name__ == "__main__":
    # Test preprocessing
    print("Testing preprocessing pipeline...")
    
    # Load sample images
    phase_both = cv2.imread('both/1_FRAME_1_PHASE_000.png')
    unstained = cv2.imread('unstained/full_agreement/normal/Process_20220322_DIC_100x_ 200 ms -IX83_D804_6893-cropped-N3.tif')
    
    # Test 'both' mode
    pipeline_both = PreprocessingPipeline(mode='both', target_size=(256, 256))
    processed_both = pipeline_both.preprocess_phase_image(phase_both)
    print(f"Processed 'both': shape={processed_both.shape}, mean={processed_both.mean():.2f}, std={processed_both.std():.2f}")
    
    # Test 'unstained' mode
    pipeline_unstained = PreprocessingPipeline(mode='unstained', target_size=(256, 256))
    processed_unstained = pipeline_unstained.preprocess_phase_image(unstained)
    print(f"Processed 'unstained': shape={processed_unstained.shape}, mean={processed_unstained.mean():.2f}, std={processed_unstained.std():.2f}")
    
    # Test augmentation
    aug = get_train_augmentation()
    augmented = aug(image=processed_both, target=processed_both)
    print(f"Augmented: image shape={augmented['image'].shape}, target shape={augmented['target'].shape}")
    
    print("\nPreprocessing pipeline test completed!")
