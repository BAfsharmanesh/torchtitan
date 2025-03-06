from typing import List
from torch import nn
import torch
from .model_profiler import ModelProfiler
from dataclasses import dataclass

@dataclass
class ModelMemoryInfo:
    model_name: str
    number_of_layers: int
    total_parameters_bytes: int
    parameters_per_layer_bytes: List[int]
    activation_parameters_bytes: List[int]
    
def get_layer_names(model: nn.Module) -> List[str]:
    """Get names of all layers or just transformer layers

    Args:
        model: PyTorch model

    Returns:
        List of layer names
    """
    names = []
    for name, child in model.named_children():
        if name == "layers":
            for name_2, child_2 in child.named_children():
                names.append(name + "." + name_2)
            continue
        # print name of the layer
        names.append(name)

    return names


def get_param_act_info(
    model_name: str,
    model: nn.Module, 
    layer_names: List[str],
    dummy_input: torch.Tensor
) -> ModelMemoryInfo:
    """Get parameter and activation memory info for model layers

    Args:
        model_name: Name of the model
        model: PyTorch model
        layer_names: List of layer names to profile
        dummy_input: Example input tensor

    Returns:
        ModelMemoryInfo containing memory statistics

    Raises:
        ValueError: If layer validation fails
    """
    profiler = ModelProfiler(model, layer_names=layer_names)
    # Get the total parameters size
    total_parameters_bytes = profiler.get_total_parameters()

    # Get the parameters size per layer
    tmp = profiler.get_parameters_per_layer()
    recorded_layer_names = [i[0] for i in tmp]
    parameters_per_layer_bytes = []
    for ln in layer_names:
        assert ln in recorded_layer_names, f"Layer {ln} not found in the model"
        parameters_per_layer_bytes.append(tmp[recorded_layer_names.index(ln)][1])

    # Get the activation size per layer
    tmp = profiler.get_activation_parameters_per_layer(dummy_input)
    recorded_layer_names = [i[0] for i in tmp]
    activation_parameters_bytes = []
    for ln in layer_names:
        assert ln in recorded_layer_names, f"Layer {ln} not found in the model"
        activation_parameters_bytes.append(tmp[recorded_layer_names.index(ln)][1])

    return ModelMemoryInfo(
        model_name=model_name,
        number_of_layers=len(layer_names),
        total_parameters_bytes=total_parameters_bytes,
        parameters_per_layer_bytes=parameters_per_layer_bytes,
        activation_parameters_bytes=activation_parameters_bytes
    )

