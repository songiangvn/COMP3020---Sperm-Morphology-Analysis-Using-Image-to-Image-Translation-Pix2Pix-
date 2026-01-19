"""
Dataset classes for virtual staining
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import json
import random

from preprocessing import PreprocessingPipeline, get_train_augmentation, get_val_augmentation, get_inference_augmentation
from preprocessing_robust import RobustPreprocessing
# NOTE: Using RobustPreprocessing for all datasets - preserves sperm details
# from preprocessing import PreprocessingPipeline, get_train_augmentation, get_val_augmentation, get_inference_augmentation
from preprocessing_robust import RobustPreprocessing


class VirtualStainingDataset(Dataset):
    """Dataset for paired phase-stain images from 'both' folder"""
    
    def __init__(self, 
                 root_dir: str,
                 transform=None,
                 mode: str = 'train'):
        """
        Args:
            root_dir: Path to 'both' folder
            transform: Albumentations transform
            mode: 'train' or 'val'
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        
        # Get all phase images
        self.phase_images = sorted([f for f in os.listdir(root_dir) if 'PHASE' in f and f.endswith('.png')])
        
        # Filter to ensure corresponding stain images exist
        self.image_pairs = []
        for phase_img in self.phase_images:
            stain_img = phase_img.replace('PHASE', 'STAIN')
            if os.path.exists(os.path.join(root_dir, stain_img)):
                self.image_pairs.append((phase_img, stain_img))
        
        print(f"Found {len(self.image_pairs)} image pairs in {root_dir}")
        
        # Use robust preprocessing for better quality
        self.preprocessor = RobustPreprocessing(target_size=(256, 256))
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        phase_name, stain_name = self.image_pairs[idx]
        
        # Load images
        phase_path = os.path.join(self.root_dir, phase_name)
        stain_path = os.path.join(self.root_dir, stain_name)
        
        phase_img = cv2.imread(phase_path)
        stain_img = cv2.imread(stain_path)
        
        # Convert BGR to RGB
        phase_img = cv2.cvtColor(phase_img, cv2.COLOR_BGR2RGB)
        stain_img = cv2.cvtColor(stain_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        phase_img = self.preprocessor.preprocess_training_image(phase_img)
        stain_img = self.preprocessor.preprocess_training_image(stain_img)
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=phase_img, target=stain_img)
            phase_img = augmented['image']
            stain_img = augmented['target']
        
        return {
            'phase': phase_img,
            'stain': stain_img,
            'phase_name': phase_name,
            'stain_name': stain_name
        }


class UnstainedInferenceDataset(Dataset):
    """Dataset for unstained images (inference only)"""
    
    def __init__(self,
                 root_dir: str,
                 transform=None,
                 recursive: bool = True):
        """
        Args:
            root_dir: Path to unstained folder (will search recursively)
            transform: Albumentations transform
            recursive: Whether to search subdirectories
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Find all image files
        self.image_paths = []
        self.relative_paths = []
        
        if recursive:
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
                for img_path in self.root_dir.rglob(ext):
                    self.image_paths.append(str(img_path))
                    self.relative_paths.append(img_path.relative_to(self.root_dir))
        else:
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
                for img_path in self.root_dir.glob(ext):
                    self.image_paths.append(str(img_path))
                    self.relative_paths.append(img_path.relative_to(self.root_dir))
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        
        # Use robust preprocessing with gentle mode for detail preservation
        self.preprocessor = RobustPreprocessing(target_size=(256, 256))
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        img_path = self.image_paths[idx]
        relative_path = self.relative_paths[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store original size for potential upscaling later
        original_size = img.shape[:2]
        
        # Preprocess with gentle mode (preserves sperm details)
        img = self.preprocessor.preprocess_inference_image(img, preserve_details=True)
        
        # Apply transform
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        return {
            'image': img,
            'path': str(relative_path),
            'original_size': original_size
        }


def create_train_val_split(root_dir: str, 
                           val_ratio: float = 0.15,
                           seed: int = 42) -> Tuple[List[str], List[str]]:
    """Create train/val split from image pairs"""
    np.random.seed(seed)
    
    # Get all phase images
    phase_images = sorted([f for f in os.listdir(root_dir) if 'PHASE' in f and f.endswith('.png')])
    
    # Filter to ensure corresponding stain images exist
    valid_pairs = []
    for phase_img in phase_images:
        stain_img = phase_img.replace('PHASE', 'STAIN')
        if os.path.exists(os.path.join(root_dir, stain_img)):
            valid_pairs.append(phase_img)
    
    # Shuffle and split
    np.random.shuffle(valid_pairs)
    n_val = int(len(valid_pairs) * val_ratio)
    
    val_images = valid_pairs[:n_val]
    train_images = valid_pairs[n_val:]
    
    return train_images, val_images


class SplitVirtualStainingDataset(Dataset):
    """Dataset with pre-defined train/val split"""
    
    def __init__(self,
                 root_dir: str,
                 image_list: List[str],
                 transform=None):
        """
        Args:
            root_dir: Path to 'both' folder
            image_list: List of phase image filenames to include
            transform: Albumentations transform
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_list = image_list
        
        # Create pairs
        self.image_pairs = []
        for phase_img in image_list:
            stain_img = phase_img.replace('PHASE', 'STAIN')
            if os.path.exists(os.path.join(root_dir, stain_img)):
                self.image_pairs.append((phase_img, stain_img))
        
        print(f"Dataset created with {len(self.image_pairs)} pairs")
        
        # Use robust preprocessing for better quality
        self.preprocessor = RobustPreprocessing(target_size=(256, 256))
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        phase_name, stain_name = self.image_pairs[idx]
        
        # Load images
        phase_path = os.path.join(self.root_dir, phase_name)
        stain_path = os.path.join(self.root_dir, stain_name)
        
        phase_img = cv2.imread(phase_path)
        stain_img = cv2.imread(stain_path)
        
        # Convert BGR to RGB
        phase_img = cv2.cvtColor(phase_img, cv2.COLOR_BGR2RGB)
        stain_img = cv2.cvtColor(stain_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        phase_img = self.preprocessor.preprocess_training_image(phase_img)
        stain_img = self.preprocessor.preprocess_training_image(stain_img)
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=phase_img, target=stain_img)
            phase_img = augmented['image']
            stain_img = augmented['target']
        
        return {
            'phase': phase_img,
            'stain': stain_img,
            'phase_name': phase_name,
            'stain_name': stain_name
        }


class AugmentedVirtualStainingDataset(SplitVirtualStainingDataset):
    """Dataset with domain adaptation augmentation (background injection)"""
    
    def __init__(self,
                 root_dir: str,
                 image_list: List[str],
                 background_dir: Optional[str] = None,
                 transform=None,
                 augment_prob: float = 0.8,
                 bg_blend_alpha: float = 0.3,
                 invert_prob: float = 0.5,
                 lift_black_range: Tuple[int, int] = (30, 80),
                 noise_sigma_range: Tuple[int, int] = (5, 15),
                 contrast_reduction_range: Tuple[float, float] = (0.5, 0.8)):
        """
        Args:
            root_dir: Path to 'both' folder
            image_list: List of phase image filenames
            background_dir: Directory with background patches for injection
            transform: Albumentations transform
            augment_prob: Probability of applying augmentation
            bg_blend_alpha: Weight for background blending
            invert_prob: Probability of inverting image
            lift_black_range: Range for lifting black level
            noise_sigma_range: Range for Gaussian noise sigma
            contrast_reduction_range: Range for contrast reduction
        """
        super().__init__(root_dir, image_list, transform)
        
        self.augment_prob = augment_prob
        self.bg_blend_alpha = bg_blend_alpha
        self.invert_prob = invert_prob
        self.lift_black_range = lift_black_range
        self.noise_sigma_range = noise_sigma_range
        self.contrast_reduction_range = contrast_reduction_range
        
        # Load background patches
        self.background_patches = []
        if background_dir is not None:
            bg_path = Path(background_dir)
            if bg_path.exists():
                for bg_file in bg_path.glob('*.tif'):
                    bg_img = cv2.imread(str(bg_file), cv2.IMREAD_UNCHANGED)
                    if bg_img is not None:
                        self.background_patches.append(bg_img)
                print(f"Loaded {len(self.background_patches)} background patches for augmentation")
    
    def apply_background_injection(self, img: np.ndarray) -> np.ndarray:
        """Blend random background patch onto image"""
        if len(self.background_patches) == 0:
            return img
        
        # Get random background
        bg_patch = random.choice(self.background_patches).copy().astype(np.float32)
        
        # Normalize background to [0, 255]
        bg_min, bg_max = bg_patch.min(), bg_patch.max()
        if bg_max > bg_min:
            bg_patch = (bg_patch - bg_min) / (bg_max - bg_min) * 255.0
        
        # Resize to match image
        h, w = img.shape[:2]
        bg_patch = cv2.resize(bg_patch, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Match channels
        if len(img.shape) == 3 and len(bg_patch.shape) == 2:
            bg_patch = cv2.cvtColor(bg_patch.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 2 and len(bg_patch.shape) == 3:
            bg_patch = cv2.cvtColor(bg_patch.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Blend
        blended = cv2.addWeighted(
            img.astype(np.float32), 1.0,
            bg_patch.astype(np.float32), self.bg_blend_alpha,
            0
        )
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def apply_intensity_transforms(self, img: np.ndarray) -> np.ndarray:
        """Apply intensity transformations to simulate target domain"""
        img = img.astype(np.float32)
        
        # Invert
        if random.random() < self.invert_prob:
            img = 255.0 - img
        
        # Lift black level
        lift = random.uniform(*self.lift_black_range)
        img = img + lift
        
        # Add Gaussian noise
        noise_sigma = random.uniform(*self.noise_sigma_range)
        noise = np.random.normal(0, noise_sigma, img.shape)
        img = img + noise
        
        # Reduce contrast
        mean = img.mean()
        contrast_factor = random.uniform(*self.contrast_reduction_range)
        img = mean + (img - mean) * contrast_factor
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        phase_name, stain_name = self.image_pairs[idx]
        
        # Load images
        phase_path = os.path.join(self.root_dir, phase_name)
        stain_path = os.path.join(self.root_dir, stain_name)
        
        phase_img = cv2.imread(phase_path)
        stain_img = cv2.imread(stain_path)
        
        # Convert BGR to RGB
        phase_img = cv2.cvtColor(phase_img, cv2.COLOR_BGR2RGB)
        stain_img = cv2.cvtColor(stain_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        phase_img = self.preprocessor.preprocess_training_image(phase_img)
        stain_img = self.preprocessor.preprocess_training_image(stain_img)
        
        # Apply domain adaptation augmentation with probability
        if random.random() < self.augment_prob:
            # Background injection
            if len(self.background_patches) > 0:
                phase_img = self.apply_background_injection(phase_img)
            
            # Intensity transforms
            phase_img = self.apply_intensity_transforms(phase_img)
        
        # Apply standard augmentation
        if self.transform:
            augmented = self.transform(image=phase_img, target=stain_img)
            phase_img = augmented['image']
            stain_img = augmented['target']
        
        return {
            'phase': phase_img,
            'stain': stain_img,
            'phase_name': phase_name,
            'stain_name': stain_name
        }


def get_dataloaders(root_dir: str,
                   batch_size: int = 16,
                   num_workers: int = 4,
                   val_ratio: float = 0.15,
                   seed: int = 42,
                   use_augmentation: bool = False,
                   background_dir: Optional[str] = None,
                   augment_prob: float = 0.8,
                   bg_blend_alpha: float = 0.3,
                   invert_prob: float = 0.5,
                   lift_black_range: Tuple[int, int] = (30, 80),
                   noise_sigma_range: Tuple[int, int] = (5, 15),
                   contrast_reduction_range: Tuple[float, float] = (0.5, 0.8)) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders
    
    Args:
        root_dir: Path to data directory
        batch_size: Batch size
        num_workers: Number of workers
        val_ratio: Validation split ratio
        seed: Random seed
        use_augmentation: Whether to use domain adaptation augmentation
        background_dir: Directory with background patches (required if use_augmentation=True)
        augment_prob: Probability of applying augmentation (default 0.8)
        bg_blend_alpha: Background blend weight (default 0.3)
        invert_prob: Probability of inverting image (default 0.5)
        lift_black_range: Range for lifting black level (default (30, 80))
        noise_sigma_range: Range for Gaussian noise sigma (default (5, 15))
        contrast_reduction_range: Range for contrast reduction (default (0.5, 0.8))
    """
    
    # Create split
    train_list, val_list = create_train_val_split(root_dir, val_ratio, seed)
    
    # Create datasets
    if use_augmentation:
        if background_dir is None:
            raise ValueError("background_dir must be provided when use_augmentation=True")
        
        train_dataset = AugmentedVirtualStainingDataset(
            root_dir=root_dir,
            image_list=train_list,
            background_dir=background_dir,
            transform=get_train_augmentation(),
            augment_prob=augment_prob,
            bg_blend_alpha=bg_blend_alpha,
            invert_prob=invert_prob,
            lift_black_range=lift_black_range,
            noise_sigma_range=noise_sigma_range,
            contrast_reduction_range=contrast_reduction_range
        )
    else:
        train_dataset = SplitVirtualStainingDataset(
            root_dir=root_dir,
            image_list=train_list,
            transform=get_train_augmentation()
        )
    
    val_dataset = SplitVirtualStainingDataset(
        root_dir=root_dir,
        image_list=val_list,
        transform=get_val_augmentation()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing dataset...")
    
    # Test VirtualStainingDataset
    train_transform = get_train_augmentation()
    dataset = VirtualStainingDataset(
        root_dir='both',
        transform=train_transform,
        mode='train'
    )
    
    print(f"\nDataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Phase shape: {sample['phase'].shape}")
    print(f"Stain shape: {sample['stain'].shape}")
    
    # Test dataloaders
    print("\nTesting dataloaders...")
    train_loader, val_loader = get_dataloaders('both', batch_size=8)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    batch = next(iter(train_loader))
    print(f"Batch phase shape: {batch['phase'].shape}")
    print(f"Batch stain shape: {batch['stain'].shape}")
    
    # Test UnstainedInferenceDataset
    print("\nTesting UnstainedInferenceDataset...")
    inference_dataset = UnstainedInferenceDataset(
        root_dir='unstained',
        transform=get_inference_augmentation()
    )
    print(f"Inference dataset size: {len(inference_dataset)}")
    if len(inference_dataset) > 0:
        sample = inference_dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Path: {sample['path']}")
    
    print("\nDataset test completed!")
