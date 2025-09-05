from torchvision import transforms
from PIL import Image

def load_and_preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # resizes to fixed size
        transforms.ToTensor()           # converts to tensor in [0,1]
    ])
    img = Image.open(image_path).convert("RGB")  # ensure 3 channels
    return transform(img).unsqueeze(0)           # add batch dim
