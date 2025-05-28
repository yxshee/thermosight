"""
make_dataset.py: Load raw images, apply transforms, split, and save processed data.
"""
import os
import argparse
import shutil
import random
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def build_dataloaders(input_dir, output_dir, img_size, batch_size, split_ratio=0.8, num_workers=4):
    """
    Read raw images, resize, save processed splits, then return DataLoaders.
    """
    # Define normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Prepare processed dirs
    for split in ['train', 'test']:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir)
    # Collect classes
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    pil_transform = transforms.Resize((img_size, img_size))
    for cls in classes:
        cls_input = os.path.join(input_dir, cls)
        images = [f for f in os.listdir(cls_input) if os.path.isfile(os.path.join(cls_input, f))]
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        splits = {'train': images[:split_idx], 'test': images[split_idx:]}
        for split, imgs in splits.items():
            cls_output = os.path.join(output_dir, split, cls)
            os.makedirs(cls_output, exist_ok=True)
            for img_name in imgs:
                src = os.path.join(cls_input, img_name)
                dst = os.path.join(cls_output, img_name)
                img = Image.open(src).convert('RGB')
                img = pil_transform(img)
                img.save(dst)
    # Create datasets and loaders
    train_ds = ImageFolder(os.path.join(output_dir, 'train'), transform=transforms.Compose([
        transforms.ToTensor(), normalize
    ]))
    test_ds = ImageFolder(os.path.join(output_dir, 'test'), transform=transforms.Compose([
        transforms.ToTensor(), normalize
    ]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create processed dataset')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=460)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Example call
    build_dataloaders(args.input_dir, args.output_dir, args.img_size, args.batch_size)
    print("Data preparation complete.")
