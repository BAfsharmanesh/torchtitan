from dataclasses import dataclass
from typing import List, Dict, Optional
import torch

@dataclass
class TensorMemoryInfo:
    storage_id: int  # unique storage id of the tensor
    layer_name: str  # name of the layer that the tensor belongs to
    param_num: Optional[int] = None  # parameter index in model.parameters()

@dataclass
class ModelMemoryInfo:
    model_name: str
    number_of_layers: int
    total_parameters_bytes: int
    parameters_per_layer_bytes: List[int]
    activation_parameters_bytes: List[int]

@dataclass
class LayerMemoryMetrics:
    activation_memory: float  # in MB
    weight_memory: float     # in MB
    grad_memory: float       # in MB
    optimizer_memory: float  # in MB

@dataclass
class MemoryUsageMetrics:
    activation: Dict[str, List[float]]
    weight: Dict[str, List[float]]
    grad: Dict[str, List[float]]
    optimizer: Dict[str, List[float]]
    total: Dict[str, List[float]]
    layer_memory_total_mb: List[float]
    