import torch
from torch.utils.data import DataLoader, Dataset


class RandomMoEDataset(Dataset):
    """
    A synthetic dataset for MultiLayerHierarchicalMoE.
    Generates random input data with a shape matching the model's hidden size.
    """

    def __init__(self, num_samples, seq_len, input_dim, num_classes):
        self.num_samples = num_samples
        self.data = torch.randn(
            (num_samples, seq_len, input_dim)
        )  # Random input features
        self.labels = torch.randint(
            0, num_classes, (num_samples,)
        )  # Random class labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def build_moe_data_loader(batch_size: int, seq_len: int, input_dim: int, num_classes: int) -> DataLoader:
    """
    Builds a DataLoader for the RandomMoEDataset.
    """
    dataset = RandomMoEDataset(
        num_samples=1000,
        seq_len=seq_len,
        input_dim=input_dim,
        num_classes=num_classes,
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    return data_loader
