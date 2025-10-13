# data_utils.py

import os
import shutil
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config

def setup_directories():
    """Creates the necessary train, validation, and test directories."""
    print("Setting up data directories...")
    class_names = [d for d in os.listdir(config.ORIGINAL_DATA_DIR) if os.path.isdir(os.path.join(config.ORIGINAL_DATA_DIR, d))]
    for split in ['train', 'val', 'test']:
        for class_name in class_names:
            os.makedirs(os.path.join(config.BASE_DATA_DIR, split, class_name), exist_ok=True)
    print("Directory setup complete.")
    return class_names

def prepare_and_split_data():
    """Downloads data, performs a stratified split, and copies files."""
    # Download and copy data to a writable directory
    print("Downloading dataset...")
    path = kagglehub.dataset_download(config.DATASET_URL)
    shutil.copytree(path, config.ORIGINAL_DATA_DIR, dirs_exist_ok=True)
    print(f"Dataset copied to {config.ORIGINAL_DATA_DIR}")

    class_names = setup_directories()

    # Gather all file paths and labels
    all_files = []
    for class_name in class_names:
        class_dir = os.path.join(config.ORIGINAL_DATA_DIR, class_name)
        for fname in os.listdir(class_dir):
            if os.path.isfile(os.path.join(class_dir, fname)):
                all_files.append({'path': os.path.join(class_dir, fname), 'label': class_name})

    df = pd.DataFrame(all_files)

    # Stratified split
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=config.SEED, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=config.SEED, stratify=temp_df['label'])

    # Copy files
    for split_df, split_name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        for _, row in split_df.iterrows():
            src_path = row['path']
            dst_path = os.path.join(config.BASE_DATA_DIR, split_name, row['label'], os.path.basename(src_path))
            shutil.copy(src_path, dst_path)
        print(f"Finished copying {split_name} files.")

    return len(class_names)

def get_dataloaders():
    """Returns train, validation, and test dataloaders."""
    train_tfms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.shape[0]==1 else t),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.expand(3, -1, -1) if t.shape[0]==1 else t),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(config.BASE_DATA_DIR, 'train'), transform=train_tfms)
    val_ds = datasets.ImageFolder(os.path.join(config.BASE_DATA_DIR, 'val'), transform=val_tfms)
    test_ds = datasets.ImageFolder(os.path.join(config.BASE_DATA_DIR, 'test'), transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    print("DataLoaders are ready.")
    return train_loader, val_loader, test_loader, train_ds.classes