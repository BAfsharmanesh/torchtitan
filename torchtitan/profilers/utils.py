from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import asdict
from pathlib import Path
from .constants import ACTIVATION_SAFETY_FACTOR, WEIGHT_SAFETY_FACTOR, MEMORY_SAFTEY_FACTOR
import torch
from .types import ModelMemoryInfo, ModelMetrics, Parameters, Model, ExecutionTime, ExecutionMemory


def create_model_metrics(
    time_profile: dict,
    memory_profile: dict,
    model_profile: ModelMemoryInfo,
) -> ModelMetrics:
    """Create ModelMetrics instance from profile data."""
    model_metrics = ModelMetrics(
        model=Model(
            model_name=model_profile.model_name,
            num_layers=model_profile.number_of_layers,
            parameters=Parameters(
                total_parameters_bytes=model_profile.total_parameters_bytes,
                parameters_per_layer_bytes=model_profile.parameters_per_layer_bytes,
                activation_parameters_bytes=model_profile.activation_parameters_bytes,
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
    return model_metrics

def match_partial_profiled_to_full_model(
    original_metrics: list,
    num_layers_total: int,
    num_layers_profiled: int,
    first_layer_idx: int
) -> tuple[list, float]:
    """Extrapolate profiled layer metrics to match the full model size.

    This function takes metrics from a partially profiled model and extends them
    to match the full model size by:
    1. Keeping the first few layers as is (up to first_layer_idx)
    2. Computing average of profiled layers and repeating it for middle layers
    3. Keeping the remaining layers as is

    Args:
        original_metrics: Original list of metrics per layer
        num_layers_total: Total number of layers in the full model
        num_layers_profiled: Number of layers that were actually profiled
        first_layer_idx: Index of the first layer to start averaging from

    Returns:
        tuple[list, float]: (
            Extended metrics list matching full model size,
            Average value used for extension
        )

    Example:
        If original_metrics=[x1,x2,x3,x4], first_layer_idx=1, num_layers_total=4:
        Returns: [x1,x2,x2,x2,x2,x3,x4], avg(x2)
    """
    extended_metrics = []
    
    # Copy initial layers unchanged
    for i in range(first_layer_idx):
        extended_metrics.append(original_metrics[i])
    
    # Calculate average of profiled layers
    profiled_layers = original_metrics[first_layer_idx : first_layer_idx + num_layers_profiled]
    avg_metric = sum(profiled_layers) / num_layers_profiled
    
    # Extend middle section with averaged value
    for _ in range(num_layers_total):
        extended_metrics.append(avg_metric)
    
    # Copy remaining layers unchanged
    remaining_start = first_layer_idx + num_layers_profiled
    extended_metrics.extend(original_metrics[remaining_start:])
    
    return extended_metrics, avg_metric

def update_partial_metrics_for_full_model(
    metrics: ModelMetrics,
    actual_layers: tuple[int, int],
    first_layer_index: int
) -> ModelMetrics:
    """Update metrics to match full model size."""

    actual_n_layers = actual_layers[0]
    profiled_n_layers = actual_layers[1]

    # update model parameters
    metrics.model.num_layers = actual_layers[0]
    tmp = metrics.model.parameters.parameters_per_layer_bytes

    tmp2, avg2 = match_partial_profiled_to_full_model(
        tmp, actual_n_layers, profiled_n_layers, first_layer_index
    )
    metrics.model.parameters.parameters_per_layer_bytes = tmp2

    metrics.model.parameters.total_parameters_bytes = sum(tmp2)

    tmp = metrics.model.parameters.activation_parameters_bytes
    tmp2, avg2 = match_partial_profiled_to_full_model(
        tmp, actual_n_layers, profiled_n_layers, first_layer_index
    )
    metrics.model.parameters.activation_parameters_bytes = tmp2

    # update execution time
    tmp = metrics.execution_time.layer_compute_total_ms
    sum_prev_layer_compute = sum(tmp)
    tmp2, avg_prev_layer_compute = match_partial_profiled_to_full_model(
        tmp, actual_n_layers, profiled_n_layers, first_layer_index
    )
    metrics.execution_time.layer_compute_total_ms = tmp2

    metrics.execution_time.forward_backward_time_ms += (
        avg_prev_layer_compute * (actual_n_layers - profiled_n_layers)
    )

    prev_optimizer_time = metrics.execution_time.optimizer_time_ms

    metrics.execution_time.optimizer_time_ms = (
        prev_optimizer_time
        + prev_optimizer_time
        * (avg_prev_layer_compute / sum_prev_layer_compute)
        * (actual_n_layers - profiled_n_layers)
    )

    metrics.execution_time.total_time_ms += (
        metrics.execution_time.optimizer_time_ms
        - prev_optimizer_time
        + avg_prev_layer_compute * (actual_n_layers - profiled_n_layers)
    )
    
    # update execution memory
    tmp = metrics.execution_memory.layer_memory_total_mb
    tmp2, avg2 = match_partial_profiled_to_full_model(
        tmp, actual_n_layers, profiled_n_layers, first_layer_index
    )
    metrics.execution_memory.layer_memory_total_mb = tmp2

    metrics.execution_memory.total_memory_mb += avg2 * (
        actual_n_layers - profiled_n_layers
    )
    
    return metrics


def save_metrics_to_file(
    metrics: ModelMetrics,
    file_path: str,
    device: str,
    tp: int,
    bs: int,
    rank: Optional[int] = None
) -> None:
    """Save metrics to JSON file."""
    model_metrics_json = json.dumps(asdict(metrics), indent=2)
    model_name = metrics.model.model_name
    # save file to file_path/DeviceType.{device}_{rank}_tp{tp}_bs{bs}.json
    if rank is not None:
        rank = f"_{rank}"
    else:
        rank = ""
    file_path = (
        Path(file_path)
        / f"{model_name}_DeviceType.{device}{rank}_tp{tp}_bs{bs}.json"
    )
    with open(file_path.absolute(), "w") as f:
        f.write(model_metrics_json)
    
def save_metrics(
    time_profile: dict,
    memory_profile: dict,
    model_profile: dict,
    file_path: str,
    tp: int,
    bs: int,
    device: str,
    actual_profiler_number_of_layers: Optional[tuple[int, int]] = None,
    first_layer_index: Optional[int] = None,
    rank: Optional[int] = None,
) -> str:
    """Save model profiling metrics to JSON file.
    
    Args:
        time_profile: Time profiling results
        memory_profile: Memory profiling results
        model_profile: Model architecture profile
        file_path: Output directory path
        tp: Tensor parallel degree
        bs: Batch size
        device: Device type
        actual_profiler_number_of_layers: Tuple of (actual layers, profiled layers)
        first_layer_index: Index of first layer
        rank: Process rank for distributed training
        
    Returns:
        JSON string of metrics
    """        
    # apply tp on the memory profile
    memory_profile["total"]["total_memory"] *= tp
    memory_profile["layer_memory_total_mb"] = [
        memory * tp for memory in memory_profile["layer_memory_total_mb"]
    ]
    
    # Create model metrics
    metrics = create_model_metrics(time_profile, memory_profile, model_profile)


    if actual_profiler_number_of_layers is not None:
        metrics = update_partial_metrics_for_full_model(
            metrics,
            actual_profiler_number_of_layers,
            first_layer_index,
        )


    save_metrics_to_file(metrics, file_path, device, tp, bs, rank)

    return json.dumps(asdict(metrics), indent=2)


def get_dummy_input(config, model_config):
    # Prepare a dummy input based on the specified input size
    if config.model.name == "llama2":
        dummy_input = torch.randint(
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
        dummy_input = torch.empty(
            *batch_input_size, device="meta", dtype=torch.float32
        )

    elif config.model.name == "wideresnet":
        batch_input_size = (
            config.training.batch_size,
            model_config.input_channels,
            model_config.input_size,
            model_config.input_size,
        )
        dummy_input = torch.empty(
            *batch_input_size, device="meta", dtype=torch.float32
        )
    else:
        raise ValueError(f"Unsupported model name: {config.model.name}")
    return dummy_input

def slice_layers_2_fit_gpu(act_weight_profiled, gpu_memory, tp_degree):
    
    model_name = act_weight_profiled.model_name
    model_name = "_".join(model_name.split('_')[:-1])
            
    gpu_memory = float(gpu_memory) / MEMORY_SAFTEY_FACTOR[model_name]
    
    # print(f"GPU Memory: {gpu_memory}")
    parameters_per_layer_bytes = act_weight_profiled.parameters_per_layer_bytes
    activation_parameters_bytes = act_weight_profiled.activation_parameters_bytes

    assert len(parameters_per_layer_bytes) == len(
        activation_parameters_bytes
    ), "Number of layers mismatch"
    number_of_layers = len(parameters_per_layer_bytes)

    # predict memory usage for each layer
    def _memory_usage_precidtions(weight, act, model_name):
        return (ACTIVATION_SAFETY_FACTOR[model_name]*act + WEIGHT_SAFETY_FACTOR[model_name] * weight * 4)/tp_degree



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