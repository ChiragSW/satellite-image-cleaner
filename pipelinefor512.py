import torch
from torchvision import transforms
from PIL import Image
import os
from models.cloud_classifier import CloudClassifier
from models.esrgan import RRDBNet
from models.partialconv import PartialConvInpaint
from models.cloud_segmenter import CloudSegmenterUNet # <-- Import the new model

# --- Robust Path Calculation ---
# This ensures the script finds the checkpoint files no matter where you run it from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPAINTING_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "gated_conv_inpainter.pth")
CLASSIFIER_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "cloud_classifier_best.pth")
ESRGAN_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "RRDB_ESRGAN_x4.pth")
SEGMENTER_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "cloud_segmenter.pth") 
# --- End of Path Calculation ---


class EnhancementPipeline:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. Cloud Classifier (To decide if an image needs processing)
        self.cloud_model = CloudClassifier(num_classes=2)
        self.cloud_model.load_state_dict(
            torch.load(CLASSIFIER_CHECKPOINT, map_location=self.device)
        )
        self.cloud_model.to(self.device).eval()

        # 2. Cloud Segmenter (To find the *exact* location of clouds)
        self.segmenter_model = CloudSegmenterUNet()
        if os.path.exists(SEGMENTER_CHECKPOINT):
            print(f"Loading Cloud Segmenter from {SEGMENTER_CHECKPOINT}")
            # Handle the checkpoint format from our training script
            ckpt = torch.load(SEGMENTER_CHECKPOINT, map_location=self.device)
            if 'model_state_dict' in ckpt:
                 self.segmenter_model.load_state_dict(ckpt['model_state_dict'])
            else: # For older/simpler checkpoint files
                 self.segmenter_model.load_state_dict(ckpt)
        else:
            print("WARNING: Cloud segmenter checkpoint not found. The pipeline will not be able to inpaint clouds precisely.")
        self.segmenter_model.to(self.device).eval()

        # 3. ESRGAN (To enhance resolution)
        self.esrgan = RRDBNet(scale=4)
        self.esrgan.load_state_dict(
            torch.load(ESRGAN_CHECKPOINT, map_location=self.device),
            strict=False
        )
        self.esrgan.to(self.device).eval()

        # 4. GatedConv Inpainter (To remove the clouds found by the segmenter)
        self.inpaint_model = PartialConvInpaint(
            checkpoint_path=INPAINTING_CHECKPOINT,
            device=self.device
        )

        # Transform for the initial classification step
        self.classifier_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transform for the main enhancement/inpainting path
        self.enhance_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def classify(self, img_tensor):
        with torch.no_grad():
            logits = self.cloud_model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            # The original classifier was trained with 0=cloudy, 1=clear.
            pred_class = "Cloudy" if pred_idx == 0 else "Clear"
        return pred_class, probs.squeeze().cpu().numpy()

    def enhance(self, img_tensor, pred):
        # The image tensor is in [0, 1] range here
        if pred == "Cloudy":
            print("Cloud detected. Generating precise mask with Segmenter...")
            # Use the segmenter to create a precise mask of the clouds
            with torch.no_grad():
                # The segmenter U-Net was trained on normalized images
                normalized_for_seg = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor.squeeze(0)).unsqueeze(0)
                generated_mask = self.segmenter_model(normalized_for_seg)
                # Convert the soft mask (0->1) to a hard binary mask (0 or 1)
                binary_mask = (generated_mask > 0.5).float()

            print("Inpainting only the detected cloud regions...")
            # Normalize image from [0, 1] to [-1, 1] for the inpainter and ESRGAN models
            img_tensor_norm = img_tensor * 2.0 - 1.0
            img_decloud = self.inpaint_model(img_tensor_norm, binary_mask)
            enhanced = self.esrgan(img_decloud)
        else:
            print("Image is clear. Applying ESRGAN for super-resolution only.")
            # Normalize image from [0, 1] to [-1, 1] for ESRGAN
            img_tensor_norm = img_tensor * 2.0 - 1.0
            enhanced = self.esrgan(img_tensor_norm)
        
        # De-normalize the final output from [-1, 1] back to [0, 1]
        enhanced = (enhanced + 1) / 2.0
        return enhanced

    def run(self, image_path, save_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        print(f"--- Starting Enhancement for {os.path.basename(image_path)} ---")
        img = Image.open(image_path).convert("RGB")
        
        # Create two versions of the tensor: one for classification, one for enhancement
        img_tensor_cls = self.classifier_transform(img).unsqueeze(0).to(self.device)
        img_tensor_enh = self.enhance_transform(img).unsqueeze(0).to(self.device)
        
        # Step 1: Classify the image to see if it's cloudy
        pred, probs = self.classify(img_tensor_cls)
        print(f"Classification Result: {pred} | Probabilities: {probs}")
        
        # Step 2: Enhance the image based on the classification
        final = self.enhance(img_tensor_enh, pred)
        
        # Step 3: Save the final, high-resolution image
        out = final.squeeze().detach().cpu().clamp(0, 1)
        out_img = transforms.ToPILImage()(out)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out_img.save(save_path)
        print(f"--- Successfully saved enhanced image at {save_path} ---")

