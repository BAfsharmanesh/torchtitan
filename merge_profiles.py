import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
from torchtitan.profilers.types import Parameters, Model, ExecutionTime, ExecutionMemory, ModelMetrics

@dataclass
class ProfileData:
    rank: int
    metrics: ModelMetrics

@dataclass
class MergedMetrics:
    model: Model
    total_memory_mb: float = 0
    layer_memory_total_mb: List[float] = field(default_factory=list)
    optimizer_time_ms: float = 0
    layer_compute_total_ms: List[float] = field(default_factory=list)
    batch_generator_time_ms: float = 0
    

def json_2_model(json_data: Dict) -> ModelMetrics:
    """Convert JSON data to ModelMetrics object.
    
    Args:
        json_data: Dictionary containing model, execution time and memory data
        
    Returns:
        ModelMetrics object containing the parsed data
        
    Raises:
        KeyError: If required fields are missing from json_data
    """
    if not all(k in json_data for k in ["model", "execution_time", "execution_memory"]):
        raise KeyError("JSON data missing required fields")

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


def read_json_file(file_name: str | Path) -> Dict:
    """Read and parse a JSON file.
    
    Args:
        file_name: Path to the JSON file
        
    Returns:
        Dictionary containing the parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        JSONDecodeError: If file contains invalid JSON
    """
    with open(file_name, "r") as f:
        data = json.load(f)
    return data

def merge_execution_data(files_data: List[ProfileData]) -> MergedMetrics:
    """Merge execution data from multiple profile files.
    
    Args:
        files_data: List of ProfileData objects containing metrics from different ranks
        
    Returns:
        MergedMetrics object containing the combined data
        
    Raises:
        ValueError: If files_data is empty
    """
    if not files_data:
        raise ValueError("No profile data provided")
        
    merged = MergedMetrics(model=files_data[0].metrics.model)
    
    for profile in files_data:
        # Memory metrics
        merged.total_memory_mb += profile.metrics.execution_memory.total_memory_mb
        merged.layer_memory_total_mb += profile.metrics.execution_memory.layer_memory_total_mb
        
        # Time metrics
        merged.optimizer_time_ms += profile.metrics.execution_time.optimizer_time_ms
        merged.layer_compute_total_ms += profile.metrics.execution_time.layer_compute_total_ms

    # Average batch generator time between first and last rank
    first_time = files_data[0].metrics.execution_time.batch_generator_time_ms
    last_time = files_data[-1].metrics.execution_time.batch_generator_time_ms
    merged.batch_generator_time_ms = (first_time + last_time) / 2        
    
    return merged      



def create_final_metrics(merged: MergedMetrics) -> ModelMetrics:
    """Create final ModelMetrics from merged data.
    
    Args:
        merged: MergedMetrics object containing the combined profile data
        
    Returns:
        ModelMetrics object with final calculations
    """
    forward_backward_time_ms = sum(merged.layer_compute_total_ms)
    total_time_ms = (
        forward_backward_time_ms + merged.batch_generator_time_ms + merged.optimizer_time_ms
    )
    
    execution_time = ExecutionTime(
        total_time_ms=total_time_ms,
        forward_backward_time_ms=forward_backward_time_ms,
        batch_generator_time_ms=merged.batch_generator_time_ms,
        layernorm_grads_all_reduce_time_ms=None,
        embedding_grads_all_reduce_time_ms=None,
        optimizer_time_ms=merged.optimizer_time_ms,
        layer_compute_total_ms=merged.layer_compute_total_ms,
    )
    
    execution_memory = ExecutionMemory(
        total_memory_mb=merged.total_memory_mb,
        layer_memory_total_mb=merged.layer_memory_total_mb,
    )

    return ModelMetrics(
        model=merged.model,
        execution_time=execution_time,
        execution_memory=execution_memory,
    )
        
def merge_files(json_files: List[Path]) -> None:
    """Merge multiple JSON profile files into a single output file.
    
    Args:
        json_files: List of paths to JSON profile files
        
    Raises:
        AssertionError: If files are not in same directory or have inconsistent parameters
        ValueError: If no valid files are found
    """
    if not json_files:
        raise ValueError("No JSON files provided")
    files_data = []
    tp = None
    bs = None
    base_directory = json_files[0].parent

    # Collect and validate file data
    for i, json_file in enumerate(json_files):
        assert base_directory == json_file.parent, "All files must be in the same directory"
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
            files_data.append(ProfileData(int(rank_i), json_data))
            assert tp == tp_tmp
            assert bs == bs_tmp
            assert device_name == device_name_tmp
            assert model_name == model_name_tmp

    # Sort by rank
    files_data.sort(key=lambda x: x.rank)

    # Verify all models are identical
    model = files_data[0].metrics.model
    assert all(data.metrics.model == model for data in files_data)
    
    # Merge and create final metrics
    merged = merge_execution_data(files_data)
    final_data = create_final_metrics(merged)

    # Save merged results
    output_dir = Path(base_directory) / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{model_name}_DeviceType.{device_name}_tp{tp}_bs{bs}.json"
    with open(output_file.absolute(), "w") as f:
        json.dump(asdict(final_data), f, indent=2)
    print(f"Saved merged file: {output_file}")

def parse_file_name(file_name: str) -> Tuple[List[str], str, str, str, str]:
    """Parse profile filename to extract metadata.
    
    Args:
        file_name: Name of the profile file
        
    Returns:
        Tuple containing:
            - List of run name components
            - Device name
            - Model name
            - TP (tensor parallel) size
            - Batch size
            
    Raises:
        AssertionError: If filename format is invalid
    """
    tmp = file_name.split("_DeviceType")
    if len(tmp) > 1:
        model_name, file_name = tmp
    else:
        model_name = ''
        file_name = tmp[0]
    run_name_list = file_name.split(".")[1].split("_")
    assert len(run_name_list) >= 3, f"Invalid file name: {file_name}"
    device_name = run_name_list[0]
    tp = run_name_list[-2].replace("tp", "")
    bs = run_name_list[-1].replace("bs", "")
    return run_name_list, device_name, model_name, tp, bs


def merge_all_files(base_directory: str | Path) -> None:
    """Merge all profile files in the given directory.
    
    Args:
        base_directory: Path to directory containing profile files
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    if not Path(base_directory).exists():
        raise FileNotFoundError(f"Directory not found: {base_directory}")
        
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
    base_directory = "./output_2/"
    merge_all_files(base_directory)
