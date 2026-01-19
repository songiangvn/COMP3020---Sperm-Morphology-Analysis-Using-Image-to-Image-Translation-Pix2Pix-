"""
Inference all 4 models: Pix2Pix (HoloStain), UNet, cGAN, VAE
Generate virtual stained images for classification comparison
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tensorflow as tf

# Import baseline models
from baselines.unet import UNet
from baselines.cgan import Generator as cGAN_Generator
from baselines.vae import VAE

# Import Pix2Pix model
from model import HoloStain_model


class InferenceDataset(Dataset):
    """Dataset for inference"""
    
    def __init__(self, data_dir, image_size=256):
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Get all holography images
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        
        # If concatenated, split to get holography only
        w, h = img.size
        if w > h:  # Concatenated image
            holography = img.crop((0, 0, w//2, h))
        else:  # Single image
            holography = img
        
        holography_tensor = self.transform(holography)
        
        return holography_tensor, self.image_files[idx]


def denormalize(tensor):
    """Denormalize tensor from [-1, 1] to [0, 255]"""
    tensor = tensor * 0.5 + 0.5  # [-1, 1] -> [0, 1]
    tensor = tensor.clamp(0, 1)
    tensor = tensor * 255  # [0, 1] -> [0, 255]
    return tensor.byte()


def infer_pix2pix(args):
    """Inference Pix2Pix (HoloStain) model"""
    print("\n" + "="*50)
    print("Inferring Pix2Pix (HoloStain)")
    print("="*50)
    
    output_dir = os.path.join(args.output_dir, 'pix2pix')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup TensorFlow session
    tf.compat.v1.disable_eager_execution()
    
    # Build model
    with tf.compat.v1.Session() as sess:
        # Load checkpoint
        saver = tf.compat.v1.train.import_meta_graph(os.path.join(args.pix2pix_checkpoint, 'model.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(args.pix2pix_checkpoint))
        
        # Get input/output tensors
        graph = tf.compat.v1.get_default_graph()
        input_tensor = graph.get_tensor_by_name('image_a:0')
        output_tensor = graph.get_tensor_by_name('fake_b:0')
        
        # Inference
        for img_file in tqdm(os.listdir(args.data_dir)):
            if not img_file.endswith('.png'):
                continue
            
            img_path = os.path.join(args.data_dir, img_file)
            img = Image.open(img_path).convert('RGB').resize((256, 256))
            img_array = np.array(img) / 127.5 - 1.0  # Normalize to [-1, 1]
            img_array = np.expand_dims(img_array, 0)
            
            # Generate
            output = sess.run(output_tensor, feed_dict={input_tensor: img_array})
            output = ((output[0] + 1) * 127.5).astype(np.uint8)
            
            # Save
            Image.fromarray(output).save(os.path.join(output_dir, img_file))
    
    print(f"Pix2Pix inference completed! Images saved to {output_dir}")


def infer_pytorch_model(model_type, args):
    """Inference PyTorch baseline models (UNet, cGAN, VAE)"""
    print("\n" + "="*50)
    print(f"Inferring {model_type.upper()}")
    print("="*50)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.join(args.output_dir, model_type)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    if model_type == 'unet':
        model = UNet(in_channels=3, out_channels=3).to(device)
        checkpoint_path = os.path.join(args.baseline_checkpoint, f'unet/unet_epoch_{args.epochs}.pth')
    elif model_type == 'cgan':
        model = cGAN_Generator(in_channels=3, out_channels=3).to(device)
        checkpoint_path = os.path.join(args.baseline_checkpoint, f'cgan/cgan_epoch_{args.epochs}.pth')
    elif model_type == 'vae':
        model = VAE(in_channels=3, out_channels=3, latent_dim=128).to(device)
        checkpoint_path = os.path.join(args.baseline_checkpoint, f'vae/vae_epoch_{args.epochs}.pth')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if model_type == 'cgan':
        model.load_state_dict(checkpoint['generator_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Data
    dataset = InferenceDataset(args.data_dir, args.image_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Inference
    with torch.no_grad():
        for holo, filename in tqdm(dataloader):
            holo = holo.to(device)
            
            # Generate
            if model_type == 'vae':
                output = model.generate(holo)
            else:
                output = model(holo)
            
            # Denormalize and save
            output = denormalize(output[0].cpu())
            output = output.permute(1, 2, 0).numpy()
            
            Image.fromarray(output).save(os.path.join(output_dir, filename[0]))
    
    print(f"{model_type.upper()} inference completed! Images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Inference all models')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['pix2pix', 'unet', 'cgan', 'vae', 'all'],
                        help='Model to infer (pix2pix/unet/cgan/vae/all)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing holography images to infer')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Output directory for generated images')
    parser.add_argument('--pix2pix_checkpoint', type=str, default='./output/checkpoints',
                        help='Path to Pix2Pix checkpoint directory')
    parser.add_argument('--baseline_checkpoint', type=str, default='./outputs_baselines',
                        help='Path to baseline checkpoints directory')
    parser.add_argument('--epochs', type=int, default=120,
                        help='Epoch number to load (for baselines)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    args = parser.parse_args()
    
    # Infer specified model(s)
    if args.model == 'pix2pix' or args.model == 'all':
        infer_pix2pix(args)
    
    if args.model == 'unet' or args.model == 'all':
        infer_pytorch_model('unet', args)
    
    if args.model == 'cgan' or args.model == 'all':
        infer_pytorch_model('cgan', args)
    
    if args.model == 'vae' or args.model == 'all':
        infer_pytorch_model('vae', args)
    
    print("\n" + "="*50)
    print("All inference completed!")
    print("Organize results for classification:")
    print(f"  - Pix2Pix: {os.path.join(args.output_dir, 'pix2pix')}")
    print(f"  - UNet:    {os.path.join(args.output_dir, 'unet')}")
    print(f"  - cGAN:    {os.path.join(args.output_dir, 'cgan')}")
    print(f"  - VAE:     {os.path.join(args.output_dir, 'vae')}")
    print("="*50)


if __name__ == '__main__':
    main()
