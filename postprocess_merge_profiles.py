import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
from torchtitan.profilers.types import Parameters, Model, ExecutionTime, ExecutionMemory, ModelMetrics
import re
import shutil
import os

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

def correct_memory(model_metrics: ModelMetrics) -> ModelMetrics:
    # make memory consistent with the total_memory_mb
    model_metrics.execution_memory.layer_memory_total_mb = [x * model_metrics.execution_memory.total_memory_mb / sum(model_metrics.execution_memory.layer_memory_total_mb) for x in model_metrics.execution_memory.layer_memory_total_mb]
    return model_metrics    

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
    base_directory = json_files[0].parent
    first_file = True
    device_name = None
    model_name = None
    tp = None
    bs = None

    # Collect and validate file data
    for json_file in json_files:
        assert base_directory == json_file.parent, "All files must be in the same directory"
        
        device_name_tmp, model_name_tmp, tp_tmp, bs_tmp = parse_file_name(json_file.name)
        
        if first_file:
            device_name = device_name_tmp
            model_name = model_name_tmp
            tp = tp_tmp
            bs = bs_tmp
            first_file = False
        
        # Validate consistency across files
        assert tp == tp_tmp, f"Inconsistent TP value: {tp} != {tp_tmp}"
        assert bs == bs_tmp, f"Inconsistent batch size: {bs} != {bs_tmp}"
        assert device_name == device_name_tmp, f"Inconsistent device: {device_name} != {device_name_tmp}"
        assert model_name == model_name_tmp, f"Inconsistent model: {model_name} != {model_name_tmp}"
        
        json_data = read_json_file(json_file)
        model_metrics = json_2_model(json_data)
        
        # Extract rank from filename
        rank = int(re.search(r'_(\d+)_tp\d+_bs\d+\.json$', json_file.name).group(1))
        files_data.append(ProfileData(rank, model_metrics))

    # Sort by rank
    files_data.sort(key=lambda x: x.rank)
    
    # [file.rank for file in files_data] should be [0, 1, 2, ...]
    assert all(file.rank == i for i, file in enumerate(files_data)), f"Invalid rank values for {json_files[0].name}"

    # Verify all models are identical
    model = files_data[0].metrics.model
    assert all(data.metrics.model == model for data in files_data)
    
    # Merge and create final metrics
    merged = merge_execution_data(files_data)
    final_data = create_final_metrics(merged)
    final_data = correct_memory(final_data)

    # Save merged results
    output_file_name = f"{model_name}_DeviceType.{device_name}_tp{tp}_bs{bs}.json"
    save_model_metrics(final_data, output_file_name, base_directory)
    # output_dir = Path(base_directory) / "merged"
    # output_dir.mkdir(parents=True, exist_ok=True)

    # output_file = output_dir / f"{model_name}_DeviceType.{device_name}_tp{tp}_bs{bs}.json"
    # with open(output_file.absolute(), "w") as f:
    #     json.dump(asdict(final_data), f, indent=2)
    # print(f"Saved merged file: {output_file}")


def save_model_metrics(model_metrics: ModelMetrics, output_file_name: str | Path, base_directory: Path) -> None:
    output_dir = Path(base_directory) / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / output_file_name
    with open(output_file.absolute(), "w") as f:
        json.dump(asdict(model_metrics), f, indent=2)
    print(f"Saved the file: {output_file}")


def parse_file_name(file_name: str) -> Tuple[str, str, str, str]:
    """Parse profile filename to extract metadata using regex.
    
    Args:
        file_name: Name of the profile file (format: model_name_DeviceType.device_rank_tp{N}_bs{N}.json)
        Example: "llama2_13B_DeviceType.A6000_3_tp2_bs2.json"
        
    Returns:
        Tuple containing:
            - Device name
            - Model name
            - TP (tensor parallel) size
            - Batch size
            
    Raises:
        ValueError: If filename format is invalid
    """
    pattern = r"(.+?)_DeviceType\.(\w+)_(\d+)_tp(\d+)_bs(\d+)\.json"
    match = re.match(pattern, file_name)
    
    if not match:
        raise ValueError(f"Invalid file name format: {file_name}")
        
    model_name, device_name, rank, tp, bs = match.groups()
    
    return device_name, model_name, tp, bs


def merge_all_files(base_directory: str | Path) -> None:
    """Merge all profile files in the given directory.
    
    Args:
        base_directory: Path to directory containing profile files
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    if not Path(base_directory).exists():
        raise FileNotFoundError(f"Directory not found: {base_directory}")
    
    # remove the existing merged directory
    output_dir = Path(base_directory) / "merged"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
        
    # all json files in the base_directory
    json_files = list(Path(base_directory).rglob("*.json"))


    # group json files with the same devicename, tp, bs
    json_files_grouped = {}
    for file in json_files:
        try:            
            
            # replace learning models have no rank information to the /merged directory
            pattern = r"^(.+?)_DeviceType\.([A-Za-z0-9]+)_tp(\d+)_bs(\d+)\.json$"
            if re.match(pattern, file.name):
                output_file = output_dir / file.name
                # replace if the file already exists   
                json_data = read_json_file(file)
                model_metrics = json_2_model(json_data)  
                model_metrics = correct_memory(model_metrics)                         
                save_model_metrics(model_metrics, file.name, base_directory)
                # shutil.copy2(file, output_file)
                # print(f"Copied file: {output_file}")
                continue
            
            # Skip files without rank information
            if not re.search(r'_\d+_tp\d+_bs\d+\.json$', file.name):
                continue
                
            device_name, model_name, tp, bs = parse_file_name(file.name)
            
            key = f"{model_name}_{device_name}_tp{tp}_bs{bs}"
            if key not in json_files_grouped:
                json_files_grouped[key] = []
            json_files_grouped[key].append(file)
            
        except ValueError:
            # Skip files that don't match the expected pattern
            continue

    for json_files_g in json_files_grouped.values():
        if json_files_g:
            merge_files(json_files_g)


if __name__ == "__main__":
    base_directory = "./outputs/"
    merge_all_files(base_directory)
