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

### save_metrics function dataclasses 
@dataclass
class Parameters:
    total_parameters_bytes: int
    parameters_per_layer_bytes: List[int]
    activation_parameters_bytes: List[int]

@dataclass
class Model:
    model_name: str
    num_layers: int
    parameters: Parameters

@dataclass
class ExecutionTime:
    total_time_ms: float
    forward_backward_time_ms: float
    batch_generator_time_ms: float
    layernorm_grads_all_reduce_time_ms: float
    embedding_grads_all_reduce_time_ms: float
    optimizer_time_ms: float
    layer_compute_total_ms: List[float]

@dataclass
class ExecutionMemory:
    total_memory_mb: float
    layer_memory_total_mb: List[float]

@dataclass
class ModelMetrics:
    model: Model
    execution_time: ExecutionTime
    execution_memory: ExecutionMemory
    
### save_metrics function dataclasses 
