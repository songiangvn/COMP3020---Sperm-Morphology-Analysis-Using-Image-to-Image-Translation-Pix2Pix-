#!/usr/bin/env python3
"""
Inference script for Enhanced Virtual Staining Model
Process all images in unstained folder and save results
"""
import os
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from models.enhanced_generator import EnhancedUNetGenerator
from preprocessing import PreprocessingPipeline, get_inference_augmentation
from preprocessing_robust import RobustPreprocessing


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained generator model"""
    print(f"Loading model from: {checkpoint_path}")
    
    generator = EnhancedUNetGenerator(in_channels=3, out_channels=3, base_channels=64)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'generator' in checkpoint:
        generator.load_state_dict(checkpoint['generator'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator = generator.to(device)
    generator.eval()
    
    print(f"Model loaded successfully!")
    return generator


def find_all_images(root_dir: str, extensions=['.png', '.jpg', '.tif', '.tiff']):
    """Find all images recursively"""
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(root_dir).rglob(f'*{ext}'))
    return sorted(image_paths)


def process_image(image_path: Path, generator, preprocessor, transform, device):
    """Process single image"""
    # Load image (supports 8-bit and 16-bit)
    if image_path.suffix.lower() in ['.tif', '.tiff']:
        # Use PIL for better TIFF support
        img = Image.open(str(image_path))
        image = np.array(img)
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    if image is None:
        print(f"WARNING: Failed to load {image_path}")
        return None
    
    # Preprocess with GENTLE mode (preserves sperm structure)
    image = preprocessor.preprocess_inference_image(image, preserve_details=True)
    
    # Transform
    image_tensor = transform(image=image)['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        stained_tensor = generator(image_tensor)
    
    # Post-process
    stained = stained_tensor[0].cpu()
    stained = ((stained + 1) * 127.5).clamp(0, 255)
    stained = stained.permute(1, 2, 0).numpy().astype('uint8')
    stained = cv2.cvtColor(stained, cv2.COLOR_RGB2BGR)
    
    return stained


def main():
    parser = argparse.ArgumentParser(description='Inference with Enhanced Virtual Staining')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, 
                       default='/home/23giang.ns/ML_Project/virtual_stainning/unstained',
                       help='Input directory with unstained images')
    parser.add_argument('--output_dir', type=str,
                       default='/home/23giang.ns/ML_Project/virtual_stainning/enhanced_results',
                       help='Output directory for stained images')
    parser.add_argument('--batch_process', action='store_true',
                       help='Process all images in input_dir recursively')
    parser.add_argument('--single_image', type=str, default=None,
                       help='Process single image (overrides batch_process)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    generator = load_model(args.checkpoint, device)
    
    # Setup robust preprocessing (gentle mode for detail preservation)
    preprocessor = RobustPreprocessing(target_size=(256, 256))
    transform = get_inference_augmentation()
    print("Using ROBUST preprocessing with GENTLE mode (preserves details)")
    
    # Process images
    if args.single_image:
        # Single image mode
        image_path = Path(args.single_image)
        print(f"\nProcessing single image: {image_path}")
        
        stained = process_image(image_path, generator, preprocessor, transform, device)
        
        if stained is not None:
            output_path = output_dir / f"{image_path.stem}_stained.png"
            cv2.imwrite(str(output_path), stained)
            print(f"Saved to: {output_path}")
        
    else:
        # Batch processing mode
        print(f"\nSearching for images in: {args.input_dir}")
        image_paths = find_all_images(args.input_dir)
        print(f"Found {len(image_paths)} images")
        
        if len(image_paths) == 0:
            print("No images found! Exiting.")
            return
        
        # Process all images
        success_count = 0
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                stained = process_image(image_path, generator, preprocessor, transform, device)
                
                if stained is not None:
                    # Preserve directory structure
                    relative_path = image_path.relative_to(args.input_dir)
                    output_path = output_dir / relative_path.parent / f"{relative_path.stem}_stained.png"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    cv2.imwrite(str(output_path), stained)
                    success_count += 1
            
            except Exception as e:
                print(f"\nERROR processing {image_path}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Successfully processed: {success_count}/{len(image_paths)} images")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
