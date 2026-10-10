import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from ecg_model import create_ecg_model
from data_loader import create_data_loaders


MODEL_PATH = "ecg_model.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

_, validation_loader = create_data_loaders(
    batch_size=16
)

model = create_ecg_model(
    num_classes=4
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in validation_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


class_names = validation_loader.dataset.dataset.classes


# Classification Report
print("\nClassification Report:\n")

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    zero_division=0
)

print(report)


# Confusion Matrix
cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:\n")
print(cm)


# Plot Confusion Matrix
plt.figure(figsize=(10, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    xticks_rotation=45
)

plt.title("ECG Classification Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "ecg_confusion_matrix.png",
    dpi=300
)

plt.show()

print(
    "\nConfusion matrix saved as: "
    "ecg_confusion_matrix.png"
)