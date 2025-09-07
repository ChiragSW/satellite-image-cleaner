import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

from models.cloud_segmenter import CloudSegmenterUNet

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    DATASET_DIR = "segmentation_data"
    IMAGE_DIR = os.path.join(DATASET_DIR, "images")
    MASK_DIR = os.path.join(DATASET_DIR, "masks")
    CHECKPOINT_SAVE_PATH = "checkpoints/cloud_segmenter.pth"
    EPOCHS = 50
    LR = 1e-4
    BATCH_SIZE = 4 # U-Nets can be memory intensive, start with a smaller batch size

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(MASK_DIR):
        print(f"Error: Dataset directories not found.")
        print(f"Please create '{IMAGE_DIR}' and '{MASK_DIR}' for segmentation training.")
        return

    dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CloudSegmenterUNet(n_channels=3, n_classes=1).to(device)
    # This loss is better for segmentation tasks
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_epoch = 0
    if os.path.exists(CHECKPOINT_SAVE_PATH):
        print(f"Found segmentation checkpoint! Resuming from {CHECKPOINT_SAVE_PATH}")
        checkpoint = torch.load(CHECKPOINT_SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")

    print("Starting training for the CloudSegmenterUNet...")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

        os.makedirs(os.path.dirname(CHECKPOINT_SAVE_PATH), exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, CHECKPOINT_SAVE_PATH)
        print(f"Checkpoint saved to {CHECKPOINT_SAVE_PATH}")

if __name__ == "__main__":
    main()

