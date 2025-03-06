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

    