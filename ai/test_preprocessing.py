from preprocessing import preprocess_image
import torch
import os


image_path = "../medical_reports/EMIAS-52EBC1DB_f342399e4ef745b6bfac082602f8944a.png"

image = preprocess_image(image_path)

print("Image processed successfully!")
print("Tensor shape:", image.shape)
print("Tensor type:", image.dtype)
print("File exists:", os.path.exists(image_path))
print("Is PyTorch tensor:", isinstance(image, torch.Tensor))