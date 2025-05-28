import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from src.models.vit_model import ViT

def predict_image(image_path, model_path, img_size=460, patch_size=8, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    model = ViT(img_size=img_size, patch_size=patch_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    classes = ['200°C','400°C','600°C','800°C']
    return dict(zip(classes, probs))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str)
    parser.add_argument('--model', type=str, default='models/best_model.pth')
    args = parser.parse_args()
    print(predict_image(args.image, args.model))
