import os
import torch
import torch.nn as nn
from models.partialconv_network import GatedConvUNet

class PartialConvInpaint(nn.Module):
    def __init__(self, checkpoint_path="./checkpoints/gated_conv_inpainter.pth", device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GatedConvUNet().to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading GatedConv checkpoint from {checkpoint_path}")
            # load checkpoint package
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # check if the checkpoint is a dictionary from the training script
            if 'model_state_dict' in checkpoint:
                print("Extracting model weights from training checkpoint.")
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # if not, assume it's just the model weights
                print("Loading model weights directly.")
                self.model.load_state_dict(checkpoint)
        else:
            print("No GatedConv checkpoint found. The model is UNTRAINED.")

        self.model.eval()

    def forward(self, x, mask):
        # Ensure mask has only 1 channel for concatenation
        if mask.shape[1] == 3:
            mask = mask[:, :1, :, :]

        with torch.no_grad():
            return self.model(x, mask)

