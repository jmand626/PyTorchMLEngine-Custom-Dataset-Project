import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch

from torchvision import transforms


NUM_WORKERS = 0
#NUM_WORKERS = os.cpu_count()
#doesnt work for some reason. This will certainly make it slower but I'm not sure what else I can do

# Set random seed for reproducibility
torch.manual_seed(61)
BATCH_SIZE = 48


def create_dataloaders(
        train_directory: str,
        test_directory: str,
        data_transforms: transforms.Compose,
        batch_size: int,
        workers: int = 4
):
    """
    Creates DataLoaders for training and testing datasets.

    This function takes paths to the training and testing directories,
    applies the provided transformations, and creates PyTorch DataLoaders
    for each dataset.

    Args:
        train_directory (str): Path to the training dataset directory.
        test_directory (str): Path to the testing dataset directory.
        data_transforms (torchvision.transforms.Compose): Transformations
            to apply to the images in both training and testing datasets.
        batch_size (int): Number of images per batch in each DataLoader.
        workers (int, optional): Number of subprocesses to use for data loading.
            Default is 4.

    Returns:
        tuple: (train_dataloader, test_dataloader, class_names), where
            - train_dataloader is the DataLoader for the training dataset.
            - test_dataloader is the DataLoader for the testing dataset.
            - class_names is a list of the class names in the dataset.

    Example:
        train_loader, test_loader, class_list = create_dataloaders(
            train_directory="path/to/train",
            test_directory="path/to/test",
            data_transforms=transform,
            batch_size=32,
            workers=4
        )
    """

    # Create the datasets using ImageFolder and apply the provided transforms
    train_augmented = datasets.ImageFolder(train_directory, transform=data_transforms)
    test_transformed = datasets.ImageFolder(test_directory, transform=data_transforms)

    # Retrieve class names from the training dataset
    class_names = train_augmented.classes

    # Create DataLoader for the training dataset
    train_dataloader_augmented = DataLoader(
        train_augmented,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers
    )

    # Create DataLoader for the testing dataset
    test_dataloader_simple = DataLoader(
        test_transformed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )

    return train_dataloader_augmented, test_dataloader_simple, class_names
