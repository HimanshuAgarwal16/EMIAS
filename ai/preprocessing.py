from PIL import Image
import torch
from torchvision import transforms


# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def preprocess_image(image_path: str):
    """
    Load and preprocess a medical report image.
    """

    image = Image.open(image_path)

    # Convert image to RGB
    image = image.convert("RGB")

    # Apply preprocessing
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor