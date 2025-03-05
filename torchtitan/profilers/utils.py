from typing import Dict, List, Tuple, Any, Optional
import json
import torch
from pathlib import Path
from dataclasses import dataclass, asdict
from .constants import ACTIVATION_SAFETY_FACTOR, TOTAL_SAFETY_FACTOR

                                 
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
    layernorm_grads_all_reduce_time_ms: Optional[float]
    embedding_grads_all_reduce_time_ms: Optional[float]
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

def get_dummy_input(config, model_config) -> torch.Tensor:
    """Get dummy input tensor for model profiling.
    
    Args:
        config: Training configuration object
        model_config: Model configuration object
        
    Returns:
        Dummy input tensor on meta device
    """
    if config.model.name == "llama2":
        return torch.randint(
            0,
            model_config.vocab_size,
            (config.training.batch_size, config.training.seq_len),
            device="meta",
            dtype=torch.long,
        )
    elif config.model.name == "moe":
        batch_input_size = (
            config.training.batch_size,
            config.training.seq_len,
            model_config.dim,
        )
        return torch.empty(
            *batch_input_size, device="meta", dtype=torch.float32
        )
    elif config.model.name == "wideresnet":
        batch_input_size = (
            config.training.batch_size,
            model_config.input_channels,
            model_config.input_size,
            model_config.input_size,
        )
        return torch.empty(
            *batch_input_size, device="meta", dtype=torch.float32
        )
    else:
        raise ValueError(f"Unsupported model name: {config.model.name}")

def save_metrics(
    time_profile: Dict,
    memory_profile: Dict,
    model_profile: Dict,
    file_path: str,
    tp: int,
    bs: int,
    device: str,
    actual_profiler_number_of_layers: Optional[Tuple[int, int]] = None,
    first_layer_index: Optional[int] = None,
    rank: Optional[int] = None,
) -> str:
    """Save profiling metrics to a JSON file.
    
    Args:
        time_profile: Time profiling results
        memory_profile: Memory profiling results
        model_profile: Model architecture information
        file_path: Output directory path
        tp: Tensor parallel degree
        bs: Batch size
        device: Device type
        actual_profiler_number_of_layers: Tuple of (actual layers, profiled layers)
        first_layer_index: Index of first layer
        rank: Process rank for distributed training
    
    Returns:
        JSON string containing the metrics
    """
    model_metrics = ModelMetrics(
        model=Model(
            model_name=model_profile["model_name"],
            num_layers=model_profile["number_of_layers"],
            parameters=Parameters(
                total_parameters_bytes=model_profile["total_parameters_bytes"],
                parameters_per_layer_bytes=model_profile["parameters_per_layer_bytes"],
                activation_parameters_bytes=model_profile["activation_parameters_bytes"],
            ),
        ),
        execution_time=ExecutionTime(
            total_time_ms=time_profile["total_time_ms"],
            forward_backward_time_ms=time_profile["forward_backward_time_ms"],
            batch_generator_time_ms=time_profile["batch_generator_time_ms"],
            layernorm_grads_all_reduce_time_ms=None,
            embedding_grads_all_reduce_time_ms=None,
            optimizer_time_ms=time_profile["optimizer_time_ms"],
            layer_compute_total_ms=time_profile["layer_compute_total_ms"],
        ),
        execution_memory=ExecutionMemory(
            total_memory_mb=memory_profile["total"]["total_memory"],
            layer_memory_total_mb=memory_profile["layer_memory_total_mb"],
        ),
    )

    if actual_profiler_number_of_layers:
        model_metrics = _extrapolate_metrics(
            model_metrics,
            actual_profiler_number_of_layers,
            first_layer_index
        )

    metrics_json = asdict(model_metrics)
    
    # Save to file
    rank_suffix = f"_{rank}" if rank is not None else ""
    output_path = Path(file_path) / f"{model_metrics.model.model_name}_DeviceType.{device}{rank_suffix}_tp{tp}_bs{bs}.json"
    
    with open(output_path.absolute(), "w") as f:
        json.dump(metrics_json, f, indent=2)

    return metrics_json

def _extrapolate_metrics(
    metrics: ModelMetrics,
    layer_counts: Tuple[int, int],
    first_layer_idx: int
) -> ModelMetrics:
    """Extrapolate metrics from profiled layers to full model."""
    actual_layers, profiled_layers = layer_counts
    
    def _extend_list(values: List[float]) -> Tuple[List[float], float]:
        """Extend a list of per-layer values to match actual layer count."""
        result = []
        for i in range(first_layer_idx):
            result.append(values[i])
            
        profiled_values = values[first_layer_idx:first_layer_idx + profiled_layers]
        avg_value = sum(profiled_values) / len(profiled_values)
        
        for _ in range(actual_layers):
            result.append(avg_value)
            
        result.extend(values[first_layer_idx + profiled_layers:])
        return result, avg_value

    # Update model parameters
    params = metrics.model.parameters
    extended_params, avg_params = _extend_list(params.parameters_per_layer_bytes)
    params.parameters_per_layer_bytes = extended_params
    params.total_parameters_bytes = sum(extended_params)
    
    extended_acts, _ = _extend_list(params.activation_parameters_bytes)
    params.activation_parameters_bytes = extended_acts

    # Update execution metrics
    metrics.model.num_layers = actual_layers
    
    time = metrics.execution_time
    extended_times, avg_time = _extend_list(time.layer_compute_total_ms)
    time.layer_compute_total_ms = extended_times
    time.forward_backward_time_ms += avg_time * (actual_layers - profiled_layers)
    
    prev_opt_time = time.optimizer_time_ms
    time.optimizer_time_ms = prev_opt_time * (1 + (actual_layers - profiled_layers) / profiled_layers)
    time.total_time_ms += (time.optimizer_time_ms - prev_opt_time + 
                          avg_time * (actual_layers - profiled_layers))

    # Update memory metrics
    mem = metrics.execution_memory
    extended_mem, avg_mem = _extend_list(mem.layer_memory_total_mb)
    mem.layer_memory_total_mb = extended_mem
    mem.total_memory_mb += avg_mem * (actual_layers - profiled_layers)

    return metrics 

def get_storage_size(tensor: torch.Tensor) -> int:
    """Get storage size in bytes for a tensor, handling distributed tensors."""
    if isinstance(tensor, torch.distributed.tensor.DTensor):
        return tensor.to_local().untyped_storage().nbytes()
    return tensor.untyped_storage().nbytes()

def get_storage_id(tensor: torch.Tensor) -> int:
    """Get unique storage ID for a tensor, handling distributed tensors."""
    if isinstance(tensor, torch.distributed.tensor.DTensor):
        return id(tensor.to_local().untyped_storage())
    return id(tensor.untyped_storage())

def calculate_layer_memory(
    layer: torch.nn.Module,
    include_grads: bool = True
) -> Tuple[float, float]:
    """Calculate memory usage for a layer's weights and gradients.
    
    Args:
        layer: PyTorch layer to analyze
        include_grads: Whether to include gradient memory
        
    Returns:
        Tuple of (weight_size_mb, grad_size_mb)
    """
    weight_size = sum(get_storage_size(t) for t in layer.parameters())
    grad_size = 0
    if include_grads:
        grad_size = sum(
            get_storage_size(t.grad)
            for t in layer.parameters()
            if t.grad is not None
        )
    return weight_size / (1024 * 1024), grad_size / (1024 * 1024)

def match_list_to_full_model(
    values: List[float],
    actual_layers: int,
    profiled_layers: int,
    first_layer_idx: int
) -> Tuple[List[float], float]:
    """Extend a list of per-layer values to match actual layer count.
    
    Args:
        values: Original list of values
        actual_layers: Target number of layers
        profiled_layers: Number of profiled layers
        first_layer_idx: Index of first profiled layer
        
    Returns:
        Tuple of (extended list, average value)
    """
    result = []
    # Copy initial layers
    for i in range(first_layer_idx):
        result.append(values[i])
        
    # Calculate average from profiled layers
    profiled_values = values[first_layer_idx:first_layer_idx + profiled_layers]
    avg_value = sum(profiled_values) / len(profiled_values)
    
    # Add extrapolated layers
    for _ in range(actual_layers):
        result.append(avg_value)
        
    # Copy remaining layers
    result.extend(values[first_layer_idx + profiled_layers:])
    
    return result, avg_value


def slice_layers_2_fit_gpu(act_weight_profiled, gpu_memory, tp_degree):
    # print(f"GPU Memory: {gpu_memory}")
    parameters_per_layer_bytes = act_weight_profiled["parameters_per_layer_bytes"]
    activation_parameters_bytes = act_weight_profiled["activation_parameters_bytes"]

    assert len(parameters_per_layer_bytes) == len(
        activation_parameters_bytes
    ), "Number of layers mismatch"
    number_of_layers = len(parameters_per_layer_bytes)

    # predict memory usage for each layer
    def _memory_usage_precidtions(weight, act, model_name):
        return TOTAL_SAFTEY_FACTOR[model_name]*(ACTIVATION_SAFTEY_FACTOR[model_name]*act + weight * 4)/tp_degree


    model_name = act_weight_profiled['model_name']
    model_name = "_".join(model_name.split('_')[:-1])
    # calculate size of each layer
    layer_size_tmp = []
    for i in range(number_of_layers):
        layer_size_tmp.append(
            _memory_usage_precidtions(
                parameters_per_layer_bytes[i], activation_parameters_bytes[i], model_name
            )/1024/1024/1024
        )
    # print(f"{layer_size_tmp=}")
    

    # start from layer 0, append layers until memory is full, then start a new slice
    model_slices = []
    current_slice = []
    current_slice_params = 0
    current_slice_activation = 0
    for i in range(number_of_layers):
        current_slice_params += parameters_per_layer_bytes[i]
        current_slice_activation += activation_parameters_bytes[i]
        if (
            _memory_usage_precidtions(current_slice_params, current_slice_activation, model_name)
            < gpu_memory
        ):
            current_slice.append(i)
        else:
            assert len(current_slice) != 0, "Some layers are too big"
            model_slices.append(current_slice)
            current_slice = []
            current_slice_params = parameters_per_layer_bytes[i]
            current_slice_activation = activation_parameters_bytes[i]
            current_slice.append(i)

    if current_slice:
        model_slices.append(current_slice)

    # check if all layers are included, and all slices are fit in the GPU memory
    assert (
        sum([len(slice) for slice in model_slices]) == number_of_layers
    ), "Some layers are missing"
    
    model_slice_memory = []
    for ii, slice in enumerate(model_slices):
        memory_slice_usage = _memory_usage_precidtions(
                sum([parameters_per_layer_bytes[i] for i in slice]),
                sum([activation_parameters_bytes[i] for i in slice]),
                model_name,
            )
        # print(f"Slice {ii} memory usage: {memory_slice_usage/1024/1024/1024:.2f}GiB")
        model_slice_memory.append(memory_slice_usage/1024/1024/1024)
        assert (
            memory_slice_usage
            <= gpu_memory
        ), f"The slice {i} is too big"

    return model_slices, model_slice_memory 