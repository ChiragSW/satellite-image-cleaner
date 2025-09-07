import torch
from torchvision import transforms
from PIL import Image
import os
from models.cloud_classifier import CloudClassifier
from models.esrgan import RRDBNet
from models.partialconv import PartialConvInpaint

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPAINTING_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "gated_conv_inpainter.pth")
CLASSIFIER_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "cloud_classifier.pth")
ESRGAN_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "RRDB_ESRGAN_x4.pth")

class EnhancementPipeline:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cloud classifier
        self.cloud_model = CloudClassifier(num_classes=2)
        self.cloud_model.load_state_dict(
            torch.load(CLASSIFIER_CHECKPOINT, map_location=self.device)
        )
        self.cloud_model.to(self.device).eval()

        # ESRGAN super-resolution
        self.esrgan = RRDBNet(scale=4)
        self.esrgan.load_state_dict(
            torch.load(ESRGAN_CHECKPOINT, map_location=self.device),
            strict=False
        )
        self.esrgan.to(self.device).eval()

        # GatedConv inpainting - now uses the absolute path
        self.inpaint_model = PartialConvInpaint(
            checkpoint_path=INPAINTING_CHECKPOINT,
            device=self.device
        )

        # Image transforms
        self.classifier_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.enhance_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def classify(self, img_tensor):
        with torch.no_grad():
            logits = self.cloud_model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_class = "Cloudy" if pred_idx == 0 else "Clear"
        return pred_class, probs.squeeze().cpu().numpy()

    def enhance(self, img_tensor, pred):
        # Normalize for ESRGAN
        img_tensor = img_tensor * 2.0 - 1.0

        if pred == "Cloudy":
            print("Cloud detected > GatedConv > ESRGAN")
            # For GatedConv, we provide a mask of ones where clouds are
            # But since we don't have a real-time mask, we let it inpaint the whole image
            mask = torch.ones(1, 1, img_tensor.size(2), img_tensor.size(3)).to(self.device)
            img_decloud = self.inpaint_model(img_tensor, mask)
            enhanced = self.esrgan(img_decloud)
        else:
            print("Clear > ESRGAN only")
            enhanced = self.esrgan(img_tensor)
        
        # De-normalize from ESRGAN's [-1, 1] range back to [0, 1]
        enhanced = (enhanced + 1) / 2.0
        return enhanced

    def run(self, image_path, save_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")

        img_tensor_cls = self.classifier_transform(img).unsqueeze(0).to(self.device)
        img_tensor_enh = self.enhance_transform(img).unsqueeze(0).to(self.device)

        pred, probs = self.classify(img_tensor_cls)
        print(f"Prediction: {pred} | Probabilities: {probs}")

        final = self.enhance(img_tensor_enh, pred)

        out = final.squeeze().detach().cpu().clamp(0, 1)
        out_img = transforms.ToPILImage()(out)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out_img.save(save_path)
        print(f"Saved enhanced image at {save_path}")

