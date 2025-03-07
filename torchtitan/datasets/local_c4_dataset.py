import os
import json
from typing import Optional, Iterator
from pathlib import Path
import torch
from datasets import Dataset, IterableDataset
from torchtitan.logging import logger

def get_local_c4_dataset(
    dataset_path: str,
    split: str = "train",
    streaming: bool = True,
    cache_dir: Optional[str] = None
) -> Union[Dataset, IterableDataset]:
    """
    Get C4 dataset with local caching capability.
    
    Args:
        dataset_path: Original HF dataset path
        split: Dataset split (train/validation/test)
        streaming: Whether to use streaming mode
        cache_dir: Directory to store cached data
        
    Returns:
        Dataset or IterableDataset: Same output type as HF's load_dataset
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torchtitan", "datasets", "c4")
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"c4_{split}.jsonl")
    
    # If cache exists, load from it
    if os.path.exists(cache_file):
        logger.info(f"Loading C4 dataset from local cache: {cache_file}")
        if streaming:
            return IterableDataset.from_json(cache_file)
        return Dataset.from_json(cache_file)
    
    # If no cache, download and cache
    logger.info(f"Downloading C4 dataset and caching to: {cache_file}")
    from datasets import load_dataset
    
    # Download original dataset
    ds = load_dataset(dataset_path, name="en", split=split, streaming=True)
    
    # Cache the first chunk (useful for testing/development)
    CACHE_SIZE = 100_000  # Adjust based on your needs
    
    with open(cache_file, 'w') as f:
        for i, example in enumerate(ds):
            if i >= CACHE_SIZE:
                break
            json.dump(example, f)
            f.write('\n')
    
    if streaming:
        return IterableDataset.from_json(cache_file)
    return Dataset.from_json(cache_file)
