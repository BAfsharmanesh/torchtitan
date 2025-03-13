from pathlib import Path

import numpy as np
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


def custom_to_tensor(pic):
    """
    Converts a PIL image or NumPy array to a PyTorch tensor.
    - If input is a PIL image, converts it to a NumPy array first.
    - Converts NumPy array to float32 and normalizes pixel values to [0,1].
    """
    if isinstance(pic, Image.Image):  # If PIL image, convert to NumPy array
        pic = np.array(pic)

    if not isinstance(pic, np.ndarray):
        raise TypeError(f"Expected a PIL image or NumPy array, but got {type(pic)}")

    # Convert from HWC to CHW (PyTorch format) and normalize
    pic = torch.tensor(pic, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return pic


def build_wr_data_loader(batch_size: int, input_size: int) -> DataLoader:

    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            custom_to_tensor,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    ROOT_DIR = "./data"
    DOWNLOAD = not (Path(ROOT_DIR) / datasets.CIFAR100.base_folder).exists()
    data_loader = DataLoader(
        datasets.CIFAR100(
            root=ROOT_DIR, train=True, download=DOWNLOAD, transform=transform
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    return data_loader

