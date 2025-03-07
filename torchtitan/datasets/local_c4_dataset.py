import os
import json
from typing import Iterator, Dict, Any, Optional
from torch.utils.data import IterableDataset
from datasets import Dataset, load_dataset
from torchtitan.logging import logger

class LocalC4Dataset(IterableDataset):
    """Local C4 dataset that caches and streams data from disk"""
    
    def __init__(
        self,
        cache_dir: str = "~/.cache/torchtitan/datasets/c4",
        num_shards: int = 10,
        shard_size: int = 10000,
    ):
        self.cache_dir = os.path.expanduser(cache_dir)
        self.num_shards = num_shards
        self.shard_size = shard_size
        self._current_shard = 0
        self._ensure_cache()

    def _ensure_cache(self) -> None:
        """Ensures cache exists, downloads if needed"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check if we need to download
        if not self._is_cache_complete():
            logger.info("Cache incomplete, downloading C4 dataset...")
            self._download_and_cache()
        else:
            logger.info("Using cached C4 dataset")

    def _is_cache_complete(self) -> bool:
        """Checks if all expected cache files exist"""
        for i in range(self.num_shards):
            if not os.path.exists(self._get_shard_path(i)):
                return False
        return True

    def _get_shard_path(self, shard_idx: int) -> str:
        return os.path.join(self.cache_dir, f"shard_{shard_idx:05d}.jsonl")

    def _download_and_cache(self) -> None:
        """Downloads C4 dataset and caches in shards"""
        ds = load_dataset("allenai/c4", name="en", split="train", streaming=True)
        
        current_shard = []
        shard_idx = 0
        
        for item in ds:
            current_shard.append(item)
            
            if len(current_shard) >= self.shard_size:
                self._save_shard(current_shard, shard_idx)
                shard_idx += 1
                current_shard = []
                
                if shard_idx >= self.num_shards:
                    break
        
        # Save any remaining items
        if current_shard:
            self._save_shard(current_shard, shard_idx)

    def _save_shard(self, items: list, shard_idx: int) -> None:
        shard_path = self._get_shard_path(shard_idx)
        with open(shard_path, 'w') as f:
            for item in items:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved shard {shard_idx} to {shard_path}")

    def _load_shard(self, shard_idx: int) -> Iterator[Dict[str, Any]]:
        shard_path = self._get_shard_path(shard_idx)
        with open(shard_path, 'r') as f:
            for line in f:
                yield json.loads(line)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        while True:
            for shard_idx in range(self.num_shards):
                for item in self._load_shard(shard_idx):
                    yield item 