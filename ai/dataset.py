import os
from torchvision.datasets import ImageFolder
from torchvision import transforms


DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "ECG Dataset"
)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_ecg_dataset():

    dataset = ImageFolder(
        root=DATASET_PATH,
        transform=transform
    )

    return dataset