"""
make_dataset.py: Load raw images, apply transforms, split, and save processed data.
"""
import os
import argparse
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import shutil


def build_dataloaders(input_dir, output_dir, img_size, batch_size, split_ratio=0.8, num_workers=4):
    """
    Reads images from input_dir, applies transforms, and exports to output_dir.
    Also returns train and test DataLoaders.
    """
    # Placeholder for actual implementation
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(input_dir, transform=transform)
    # Split dataset
    total = len(dataset)
    train_count = int(split_ratio * total)
    test_count = total - train_count
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_count, test_count])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
