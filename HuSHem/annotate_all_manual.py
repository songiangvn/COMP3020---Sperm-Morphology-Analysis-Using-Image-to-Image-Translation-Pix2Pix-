#!/usr/bin/env python3
"""
Simple CLI annotation tool - ALL 216 images
Shows preview image, you type rotation (0/90/180/270)
Progress is auto-saved after every 10 images
"""

import os
import json
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.expanduser('~/ML_Project/HuSHem')
ANNOTATION_FILE = os.path.join(DATA_DIR, 'head_orientation_annotations.json')
PREVIEW_DIR = os.path.join(DATA_DIR, 'outputs', 'preview')
os.makedirs(PREVIEW_DIR, exist_ok=True)

# Load existing annotations
if os.path.exists(ANNOTATION_FILE):
    with open(ANNOTATION_FILE, 'r') as f:
        annotations = json.load(f)
    print(f"✓ Loaded {len(annotations)} existing annotations")
else:
    annotations = {}

# Get ALL images
classes = ['01_Normal', '02_Tapered', '03_Pyriform', '04_Amorphous']
all_images = []

for cls in classes:
    class_path = os.path.join(DATA_DIR, cls)
    if os.path.isdir(class_path):
        images = sorted([f for f in os.listdir(class_path) if f.endswith('.BMP')])
        for img_name in images:
            rel_path = os.path.join(cls, img_name)
            all_images.append(rel_path)

# Filter out already annotated (if you want to continue from where you left off)
print(f"\nTotal images: {len(all_images)}")
print(f"Already annotated: {len(annotations)}")

resume = input("\nContinue from where you left off? (y/n): ").strip().lower()
if resume == 'y':
    remaining_images = [img for img in all_images if img not in annotations]
    print(f"Remaining to annotate: {len(remaining_images)}")
else:
    remaining_images = all_images
    annotations = {}  # Start fresh
    print(f"Starting fresh annotation for all {len(all_images)} images")

if len(remaining_images) == 0:
    print("\n✓ All images already annotated!")
    exit(0)

print("\n" + "="*80)
print("ANNOTATION INSTRUCTIONS")
print("="*80)
print("For each image preview, enter the rotation needed:")
print("  0   = Head already points LEFT")
print("  90  = Rotate 90° clockwise")
print("  180 = Rotate 180° (flip)")
print("  270 = Rotate 270° clockwise (or -90°)")
print("")
print("Special commands:")
print("  r   = Re-show current image")
print("  b   = Go back to previous image")
print("  q   = Quit and save progress")
print("="*80 + "\n")

def save_progress():
    """Save annotations to file"""
    with open(ANNOTATION_FILE, 'w') as f:
        json.dump(annotations, f, indent=2)

def create_preview(img_path, rel_path, idx, total):
    """Create preview image with all 4 rotation options"""
    img = Image.open(img_path).convert('RGB')
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, rotation in enumerate([0, 90, 180, 270]):
        rotated = img.rotate(rotation, resample=Image.BICUBIC, expand=False)
        axes[i].imshow(rotated)
        axes[i].set_title(f'{rotation}°', fontsize=16, fontweight='bold')
        axes[i].axis('off')
        
        # Red arrow showing goal direction
        axes[i].annotate('', xy=(0.3, 0.95), xytext=(0.1, 0.95),
                        xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', lw=3, color='red'))
        axes[i].text(0.2, 1.0, '← HEAD', transform=axes[i].transAxes,
                    fontsize=10, color='red', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(f'[{idx+1}/{total}] {rel_path}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    preview_path = os.path.join(PREVIEW_DIR, 'current.png')
    plt.savefig(preview_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return preview_path

# History for going back
history = []

try:
    idx = 0
    while idx < len(remaining_images):
        rel_path = remaining_images[idx]
        img_path = os.path.join(DATA_DIR, rel_path)
        
        # Create and show preview
        preview_path = create_preview(img_path, rel_path, idx, len(remaining_images))
        
        print(f"\n[{idx+1}/{len(remaining_images)}] {rel_path}")
        print(f"Preview: {preview_path}")
        
        # Check if already has annotation (from previous auto-annotation)
        if rel_path in annotations:
            print(f"Current: {annotations[rel_path]}°")
        
        while True:
            response = input("Rotation (0/90/180/270) or r/b/q: ").strip().lower()
            
            if response == 'q':
                print("\n💾 Saving and quitting...")
                save_progress()
                print(f"✓ Saved {len(annotations)} annotations")
                print(f"Progress: {idx}/{len(remaining_images)} completed")
                exit(0)
            
            elif response == 'r':
                # Re-create preview (same image)
                preview_path = create_preview(img_path, rel_path, idx, len(remaining_images))
                print(f"Refreshed: {preview_path}")
                continue
            
            elif response == 'b':
                if idx > 0:
                    idx -= 1
                    print("⬅ Going back...")
                else:
                    print("Already at first image!")
                break
            
            elif response in ['0', '90', '180', '270']:
                rotation = int(response)
                annotations[rel_path] = rotation
                history.append((rel_path, rotation))
                print(f"✓ {rotation}°")
                
                # Auto-save every 10 images
                if len(annotations) % 10 == 0:
                    save_progress()
                    print(f"  [Auto-saved: {len(annotations)} annotations]")
                
                idx += 1
                break
            
            else:
                print("❌ Invalid input. Use: 0, 90, 180, 270, r, b, or q")

except KeyboardInterrupt:
    print("\n\n⚠ Interrupted by Ctrl+C")

# Final save
print("\n💾 Saving final annotations...")
save_progress()
print(f"✓ Saved {len(annotations)} annotations to: {ANNOTATION_FILE}")
print(f"\nProgress: {len(annotations)}/{len(all_images)} images annotated")
print("\nYou can resume anytime by running this script again!")
