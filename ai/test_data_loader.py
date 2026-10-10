from ai.test_data_loader import create_data_loaders


train_loader, validation_loader = create_data_loaders()

print("Data loaders created successfully!")

print(
    "Training images:",
    len(train_loader.dataset)
)

print(
    "Validation images:",
    len(validation_loader.dataset)
)

images, labels = next(iter(train_loader))

print(
    "Batch image shape:",
    images.shape
)

print(
    "Batch labels shape:",
    labels.shape
)