import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List
from torchtitan.profilers.types import Parameters, Model, ExecutionTime, ExecutionMemory, ModelMetrics



def json_2_model(json_data):
    parameters = Parameters(**json_data["model"]["parameters"])
    model = Model(
        model_name=json_data["model"]["model_name"],
        parameters=parameters,
        num_layers=json_data["model"]["num_layers"],
    )
    execution_time = ExecutionTime(**json_data["execution_time"])
    execution_memory = ExecutionMemory(**json_data["execution_memory"])
    model_metrics = ModelMetrics(
        model=model, execution_time=execution_time, execution_memory=execution_memory
    )
    return model_metrics


def read_json_file(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


def merge_files(json_files):
    files_data = []
    tp = None
    bs = None
    # get directory name
    base_directory = json_files[0].parent

    for i, json_file in enumerate(json_files):
        # assert all files have the same base_directory
        assert base_directory == json_file.parent
        run_name_list, device_name_tmp, model_name_tmp, tp_tmp, bs_tmp = (
            parse_file_name(json_file.name)
        )
        if i == 0:
            device_name = device_name_tmp
            model_name = model_name_tmp
            tp = tp_tmp
            bs = bs_tmp
        if len(run_name_list) > 3:
            json_data = read_json_file(json_file)
            json_data = json_2_model(json_data)
            rank_i = run_name_list[1]
            files_data.append((rank_i, json_data))
            assert tp == tp_tmp
            assert bs == bs_tmp
            assert device_name == device_name_tmp
            assert model_name == model_name_tmp

    # sort files_data by the first element of the tuple
    files_data.sort(key=lambda x: int(x[0]))

    # assersion that all data has same data.Model
    model = files_data[0][1].model
    for rank, data in files_data:
        assert data.model == model
    final_data_model = model

    # sum execution_memory.total_memory_mb for all ranks
    # concat all layer_memory_total_mb for all ranks
    total_memory_mb = 0
    layer_memory_total_mb = []
    for rank, data in files_data:
        total_memory_mb += data.execution_memory.total_memory_mb
        layer_memory_total_mb += data.execution_memory.layer_memory_total_mb
    final_data_execution_memory = ExecutionMemory(
        total_memory_mb=total_memory_mb, layer_memory_total_mb=layer_memory_total_mb
    )

    # sum execution_time.optimizer_time_ms for all ranks
    # concat all layer_compute_total_ms for all ranks
    # average batch_generator_time_ms for all ranks
    optimizer_time_ms = 0
    layer_compute_total_ms = []
    for rank, data in files_data:
        optimizer_time_ms += data.execution_time.optimizer_time_ms
        layer_compute_total_ms += data.execution_time.layer_compute_total_ms

    batch_generator_time_ms = files_data[0][1].execution_time.batch_generator_time_ms
    batch_generator_time_ms += files_data[-1][1].execution_time.batch_generator_time_ms
    batch_generator_time_ms /= 2

    # sum layer_compute_total_ms for forward_backward_time_ms
    forward_backward_time_ms = sum(layer_compute_total_ms)
    total_time_ms = (
        forward_backward_time_ms + batch_generator_time_ms + optimizer_time_ms
    )

    final_data_execution_time = ExecutionTime(
        total_time_ms=total_time_ms,
        forward_backward_time_ms=forward_backward_time_ms,
        batch_generator_time_ms=batch_generator_time_ms,
        layernorm_grads_all_reduce_time_ms=None,
        embedding_grads_all_reduce_time_ms=None,
        optimizer_time_ms=optimizer_time_ms,
        layer_compute_total_ms=layer_compute_total_ms,
    )

    final_data = ModelMetrics(
        model=final_data_model,
        execution_time=final_data_execution_time,
        execution_memory=final_data_execution_memory,
    )

    # save the json file
    model_metrics_json = json.dumps(asdict(final_data), indent=2)

    # save file to file_path/merged/"DeviceType.{device}_tp{tp}_bs{bs}".json
    base_directory = Path(base_directory) / "merged"
    # mkdir if not exists
    base_directory.mkdir(parents=True, exist_ok=True)

    file_path = Path(base_directory) / f"{model_name}_DeviceType.{device_name}_tp{tp}_bs{bs}.json"
    with open(file_path.absolute(), "w") as f:
        f.write(model_metrics_json)


def parse_file_name(file_name):
    tmp = file_name.split("_DeviceType")
    if len(tmp) > 1:
        model_name, file_name = tmp
    else:
        model_name = ''
        file_name = tmp[0]
    run_name_list = file_name.split(".")[1].split("_")
    assert len(run_name_list) >= 3
    device_name = run_name_list[0]
    tp = run_name_list[-2].replace("tp", "")
    bs = run_name_list[-1].replace("bs", "")
    return run_name_list, device_name, model_name, tp, bs


def merge_all_files(base_directory):
    # all json files in the base_directory
    json_files = list(Path(base_directory).rglob(f"*.json"))

    # group json files with the same devicename, tp, bs
    json_files_grouped = {}
    for file in json_files:

        run_name_list, device_name, model_name, tp, bs = parse_file_name(file.name)

        key = f"{model_name}_{device_name}_tp{tp}_bs{bs}"
        if key not in json_files_grouped:
            json_files_grouped[key] = []
        if len(run_name_list) > 3 and model_name != '':
            json_files_grouped[key].append(file)

    for json_files_g in json_files_grouped.values():
        if json_files_g:
            merge_files(json_files_g)


if __name__ == "__main__":
    base_directory = "./outputs/"
    merge_all_files(base_directory)
