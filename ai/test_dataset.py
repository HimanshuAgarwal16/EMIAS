from dataset import load_ecg_dataset


dataset = load_ecg_dataset()

print("Dataset loaded successfully!")
print("Total images:", len(dataset))
print("Classes:", dataset.classes)
print("Class mapping:", dataset.class_to_idx)