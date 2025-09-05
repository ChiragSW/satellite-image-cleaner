import torch
from models.cloud_classifier import CloudClassifier
from torchvision import transforms
from PIL import Image
import os
from models.esrgan import RRDBNet
from models.deepfill import DeepFillV2

# Placeholder ESRGAN & DeepFill
class DummyESRGAN(torch.nn.Module):
    def forward(self, x): return x

class DummyDeepFill(torch.nn.Module):
    def forward(self, x): return x

class EnhancementPipeline:
    def __init__(self, model_path="checkpoints/cloud_classifier.pth", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load classifier
        self.cloud_model = CloudClassifier(num_classes=2)
        self.cloud_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.cloud_model.to(self.device).eval()

        # Enhancement models 
        self.esrgan = RRDBNet()
        self.esrgan.load_state_dict(torch.load("checkpoints/RRDB_ESRGAN_x4.pth", map_location=self.device))
        self.esrgan.to(self.device).eval()

        self.deepfill = DeepFillV2()
        self.deepfill.load_state_dict(torch.load("checkpoints/deepfillv2_places_256x256_20200619-10d15793.pth", map_location=self.device))
        self.deepfill.to(self.device).eval()

        # Image preprocessing (resize + normalize)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # For converting tensors back to images (undo normalization)
        self.denorm = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def classify(self, img_tensor):
        """Return prediction and probability"""
        with torch.no_grad():
            logits = self.cloud_model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        return pred, probs.squeeze().cpu().numpy()

    def enhance(self, img_tensor, pred):
        """Apply conditional enhancement"""
        if pred == 1:  # Cloudy
            print("Cloud detected → DeepFill → ESRGAN")
            img_decloud = self.deepfill(img_tensor)
            return self.esrgan(img_decloud)
        else:  # Clear
            print("Clear → ESRGAN only")
            return self.esrgan(img_tensor)


    def run(self, image_path, save_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Classification
        pred, probs = self.classify(img_tensor)
        print(f"Prediction: {pred} | Probabilities: {probs}")

        # Enhancement
        final = self.enhance(img_tensor, pred)

        # Convert back to PIL
        final_img = self.denorm(final.squeeze().cpu()).clamp(0,1)
        out_img = transforms.ToPILImage()(final_img)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out_img.save(save_path)
        print(f"Saved enhanced image at {save_path}")
