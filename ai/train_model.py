import torch
import torch.nn as nn
import torch.optim as optim

from ecg_model import create_ecg_model
from data_loader import create_data_loaders


# =========================
# CONFIGURATION
# =========================

EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 16

MODEL_PATH = "ecg_model.pth"


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================
# DATA
# =========================

train_loader, validation_loader = create_data_loaders(
    batch_size=BATCH_SIZE
)


# =========================
# MODEL
# =========================

model = create_ecg_model(
    num_classes=4
)

model = model.to(device)


# =========================
# LOSS & OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# TRAINING
# =========================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = (
        100 * correct / total
    )

    train_loss = (
        running_loss /
        len(train_loader)
    )


    # =========================
    # VALIDATION
    # =========================

    model.eval()

    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            validation_total += labels.size(0)

            validation_correct += (
                predicted == labels
            ).sum().item()

    validation_accuracy = (
        100 *
        validation_correct /
        validation_total
    )


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {validation_accuracy:.2f}%"
    )


# =========================
# SAVE MODEL
# =========================

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print()
print("Training completed!")
print("Model saved as:", MODEL_PATH)