import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

# Import the new GatedConvUNet
from models.partialconv_network import GatedConvUNet

class InpaintingDataset(Dataset):
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
        mask = Image.open(mask_path).convert("L") # Grayscale mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Create the cloudy image by applying the mask
        cloudy_image = image * (1 - mask)

        return cloudy_image, mask, image # input, mask, ground_truth

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Hyperparameters ---
    DATASET_DIR = "inpainting_data"
    IMAGE_DIR = os.path.join(DATASET_DIR, "images")
    MASK_DIR = os.path.join(DATASET_DIR, "masks")
    CHECKPOINT_SAVE_PATH = "checkpoints/gated_conv_inpainter.pth"
    EPOCHS = 50
    LR = 1e-4
    BATCH_SIZE = 8

    # --- Dataset and DataLoader ---
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(MASK_DIR):
        print(f"Error: Dataset directories not found.")
        print(f"Please create '{IMAGE_DIR}' and '{MASK_DIR}' and fill them with your data.")
        return

    dataset = InpaintingDataset(IMAGE_DIR, MASK_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Model, Loss, and Optimizer ---
    model = GatedConvUNet().to(device)
    criterion = nn.L1Loss() # L1 loss is common for inpainting
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Starting training for the GatedConvUNet Inpainter...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for cloudy, mask, ground_truth in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            cloudy, mask, ground_truth = cloudy.to(device), mask.to(device), ground_truth.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(cloudy, mask)
            loss = criterion(outputs, ground_truth)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

        # Save the model checkpoint
        os.makedirs(os.path.dirname(CHECKPOINT_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), CHECKPOINT_SAVE_PATH)
        print(f"Checkpoint saved to {CHECKPOINT_SAVE_PATH}")

if __name__ == "__main__":
    main()
