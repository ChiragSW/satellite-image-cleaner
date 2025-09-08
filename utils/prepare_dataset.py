import os
import random
import shutil
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_dataset(root="data", train_ratio=0.8, exts=(".png", ".jpg", ".jpeg", ".tif")):
    random.seed(42)  # reproducibility
    
    classes = ["cloudy", "clear"]
    for cls in classes:
        input_dir = Path(root) / cls
        if not input_dir.exists():
            raise FileNotFoundError(f"Missing folder: {input_dir}")

        # Collect all images
        images = [f for f in input_dir.iterdir() if f.suffix.lower() in exts]

        # Shuffle & split
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_files = images[:split_idx]
        val_files = images[split_idx:]

        # Save into train/val
        for split, files in [("train", train_files), ("val", val_files)]:
            out_dir = Path(root) / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            for f in files:
                out_path = out_dir / (f.stem + ".png")  # normalize to PNG
                if f.suffix.lower() == ".tif":
                    img = Image.open(f).convert("RGB")
                    img.save(out_path, "PNG")
                else:
                    shutil.copy(f, out_path)

        print(f"{cls}: {len(train_files)} train, {len(val_files)} val")


def get_dataloaders(data_dir="data", image_size=224, batch_size=32, num_workers=4):
    """Build PyTorch dataloaders for train/val"""
    train_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_data = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
