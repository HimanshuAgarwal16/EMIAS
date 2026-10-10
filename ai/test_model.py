import torch
from ecg_model import create_ecg_model


model = create_ecg_model()

print("Model created successfully!")
print("Model type:", type(model).__name__)

dummy_input = torch.randn(1, 3, 224, 224)

output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)