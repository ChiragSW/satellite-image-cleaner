import torch
from torchvision import transforms
from PIL import Image
import os
from models.cloud_classifier import CloudClassifier
from models.esrgan import RRDBNet
from models.partialconv import PartialConvInpaint
from models.cloud_segmenter import CloudSegmenterUNet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPAINTING_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "gated_conv_inpainter.pth")
CLASSIFIER_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "cloud_classifier_best.pth")
ESRGAN_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "RRDB_ESRGAN_x4.pth")
SEGMENTER_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "cloud_segmenter.pth")

class EnhancementPipeline:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. Cloud Classifier
        self.cloud_model = CloudClassifier(num_classes=2)
        self.cloud_model.load_state_dict(
            torch.load(CLASSIFIER_CHECKPOINT, map_location=self.device)
        )
        self.cloud_model.to(self.device).eval()

        # 2. Cloud Segmenter
        self.segmenter_model = CloudSegmenterUNet()
        if os.path.exists(SEGMENTER_CHECKPOINT):
            print(f"Loading Cloud Segmenter from {SEGMENTER_CHECKPOINT}")
            ckpt = torch.load(SEGMENTER_CHECKPOINT, map_location=self.device)
            if 'model_state_dict' in ckpt:
                 self.segmenter_model.load_state_dict(ckpt['model_state_dict'])
            else:
                 self.segmenter_model.load_state_dict(ckpt)
        else:
            print("WARNING: Cloud segmenter checkpoint not found.")
        self.segmenter_model.to(self.device).eval()

        # 3. ESRGAN
        self.esrgan = RRDBNet(scale=4)
        self.esrgan.load_state_dict(
            torch.load(ESRGAN_CHECKPOINT, map_location=self.device),
            strict=False
        )
        self.esrgan.to(self.device).eval()

        # 4. GatedConv Inpainter
        self.inpaint_model = PartialConvInpaint(
            checkpoint_path=INPAINTING_CHECKPOINT,
            device=self.device
        )

        # Transforms
        self.classifier_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # MODIFIED: Reduced resolution to prevent Out-of-Memory errors
        self.enhance_transform = transforms.Compose([
            transforms.Resize((384, 384)),
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
        # OPTIMIZED: Wrap all inference in no_grad to save memory
        with torch.no_grad():
            if pred == "Cloudy":
                print("Cloud detected. Generating precise mask with Segmenter...")
                normalized_for_seg = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor.squeeze(0)).unsqueeze(0)
                generated_mask = self.segmenter_model(normalized_for_seg)
                binary_mask = (generated_mask > 0.5).float()

                print("Inpainting only the detected cloud regions...")
                img_tensor_norm = img_tensor * 2.0 - 1.0
                img_decloud = self.inpaint_model(img_tensor_norm, binary_mask)
                enhanced = self.esrgan(img_decloud)
            else:
                print("Image is clear. Applying ESRGAN for super-resolution only.")
                img_tensor_norm = img_tensor * 2.0 - 1.0
                enhanced = self.esrgan(img_tensor_norm)
            
            enhanced = (enhanced + 1) / 2.0
        return enhanced

    def run(self, image_path, save_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        print(f"--- Starting Enhancement for {os.path.basename(image_path)} ---")
        img = Image.open(image_path).convert("RGB")
        
        img_tensor_cls = self.classifier_transform(img).unsqueeze(0).to(self.device)
        img_tensor_enh = self.enhance_transform(img).unsqueeze(0).to(self.device)
        
        pred, probs = self.classify(img_tensor_cls)
        print(f"Classification Result: {pred} | Probabilities: {probs}")
        
        final = self.enhance(img_tensor_enh, pred)
        
        out = final.squeeze().detach().cpu().clamp(0, 1)
        out_img = transforms.ToPILImage()(out)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out_img.save(save_path)
        print(f"--- Successfully saved enhanced image at {save_path} ---")

