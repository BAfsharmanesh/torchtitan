from dataclasses import dataclass
from typing import List, Dict
import json
from dataclasses import asdict
from pathlib import Path
from .constants import ACTIVATION_SAFETY_FACTOR, TOTAL_SAFETY_FACTOR
import torch


def save_metrics(
    time_profile: dict,
    memory_profile: dict,
    model_profile: dict,
    file_path: str,
    tp: int,
    bs: int,
    device: str,
    actual_profiler_number_of_layers=None,
    first_layer_index=None,
    rank=None,
) -> Dict:

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

    model_metrics = ModelMetrics(
        model=Model(
            model_name=model_profile["model_name"],
            num_layers=model_profile["number_of_layers"],
            parameters=Parameters(
                total_parameters_bytes=model_profile["total_parameters_bytes"],
                parameters_per_layer_bytes=model_profile["parameters_per_layer_bytes"],
                activation_parameters_bytes=model_profile[
                    "activation_parameters_bytes"
                ],
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

    def match_list_to_full_model(tmp, nls, pnl, fli):
        tmp2 = []
        for i in range(fli):
            tmp2.append(tmp[i])
        avg = sum(tmp[fli : fli + pnl]) / pnl
        for _ in range(nls):
            tmp2.append(avg)
        for i in range(fli + pnl, len(tmp)):
            tmp2.append(tmp[i])

        return tmp2, avg

    if actual_profiler_number_of_layers is not None:

        # model
        actual_n_layers = actual_profiler_number_of_layers[0]
        profiled_n_layers = actual_profiler_number_of_layers[1]
        first_layer_index = first_layer_index

        model_metrics.model.num_layers = actual_profiler_number_of_layers[0]
        tmp = model_metrics.model.parameters.parameters_per_layer_bytes
        # first_layer_index = 1, actual_n_layers=4 => tmp=[x1,x2,x3,x4] , tmp2=[x1,x2,x2,x2,x2,x3,x4]
        tmp2, avg2 = match_list_to_full_model(
            tmp, actual_n_layers, profiled_n_layers, first_layer_index
        )
        model_metrics.model.parameters.parameters_per_layer_bytes = tmp2

        model_metrics.model.parameters.total_parameters_bytes = sum(tmp2)

        tmp = model_metrics.model.parameters.activation_parameters_bytes
        tmp2, avg2 = match_list_to_full_model(
            tmp, actual_n_layers, profiled_n_layers, first_layer_index
        )
        model_metrics.model.parameters.activation_parameters_bytes = tmp2

        # execution time

        tmp = model_metrics.execution_time.layer_compute_total_ms
        sum_prev_layer_compute = sum(tmp)
        tmp2, avg_prev_layer_compute = match_list_to_full_model(
            tmp, actual_n_layers, profiled_n_layers, first_layer_index
        )
        model_metrics.execution_time.layer_compute_total_ms = tmp2

        model_metrics.execution_time.forward_backward_time_ms += (
            avg_prev_layer_compute * (actual_n_layers - profiled_n_layers)
        )

        prev_optimizer_time = model_metrics.execution_time.optimizer_time_ms

        model_metrics.execution_time.optimizer_time_ms = (
            prev_optimizer_time
            + prev_optimizer_time
            * (avg_prev_layer_compute / sum_prev_layer_compute)
            * (actual_n_layers - profiled_n_layers)
        )

        model_metrics.execution_time.total_time_ms += (
            model_metrics.execution_time.optimizer_time_ms
            - prev_optimizer_time
            + avg_prev_layer_compute * (actual_n_layers - profiled_n_layers)
        )
        # execution memory

        tmp = model_metrics.execution_memory.layer_memory_total_mb
        tmp2, avg2 = match_list_to_full_model(
            tmp, actual_n_layers, profiled_n_layers, first_layer_index
        )
        model_metrics.execution_memory.layer_memory_total_mb = tmp2

        model_metrics.execution_memory.total_memory_mb += avg2 * (
            actual_n_layers - profiled_n_layers
        )

    model_metrics_json = json.dumps(asdict(model_metrics), indent=2)
    tmp_model_name = model_profile["model_name"]
    # save file to file_path/"DeviceType.{device}_tp{tp}_bs{bs}".json
    if rank is not None:
        rank = f"_{rank}"
    else:
        rank = ""
    file_path = (
        Path(file_path)
        / f"{tmp_model_name}_DeviceType.{device}{rank}_tp{tp}_bs{bs}.json"
    )
    with open(file_path.absolute(), "w") as f:
        f.write(model_metrics_json)

    return model_metrics_json


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
    # print(f"GPU Memory: {gpu_memory}")
    parameters_per_layer_bytes = act_weight_profiled["parameters_per_layer_bytes"]
    activation_parameters_bytes = act_weight_profiled["activation_parameters_bytes"]

    assert len(parameters_per_layer_bytes) == len(
        activation_parameters_bytes
    ), "Number of layers mismatch"
    number_of_layers = len(parameters_per_layer_bytes)

    # predict memory usage for each layer
    def _memory_usage_precidtions(weight, act, model_name):
        return TOTAL_SAFETY_FACTOR[model_name]*(ACTIVATION_SAFETY_FACTOR[model_name]*act + weight * 4)/tp_degree


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