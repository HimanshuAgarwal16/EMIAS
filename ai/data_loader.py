from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from dataset import load_ecg_dataset


def create_data_loaders(
    batch_size=16,
    validation_size=0.2
):

    dataset = load_ecg_dataset()

    labels = dataset.targets

    indices = list(range(len(dataset)))

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        stratify=labels,
        random_state=42
    )

    train_dataset = Subset(
        dataset,
        train_indices
    )

    validation_dataset = Subset(
        dataset,
        validation_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, validation_loader