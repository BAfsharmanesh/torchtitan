from typing import Dict, List, Tuple, Any
import torch
from .model_profiler import ModelLayerProfile
from .activation_profiler import measure_activation_shape

def get_layer_names(model: torch.nn.Module) -> List[str]:
    """Get names of all layers in model.
    
    Args:
        model: PyTorch model to inspect
        
    Returns:
        List of layer names in forward pass order
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
    model: torch.nn.Module,
    layer_names: List[str],
    dummy_input: torch.Tensor
) -> Dict[str, Any]:
    """Get parameter and activation information for model layers."""
    profiler = ModelLayerProfile(model, layer_names=layer_names)
    layer_profiles = profiler.get_layer_profiles(dummy_input)
    
    return {
        "model_name": model_name,
        "number_of_layers": len(layer_profiles),
        "total_parameters_bytes": sum(p.parameter_size for p in layer_profiles),
        "parameters_per_layer_bytes": [p.parameter_size for p in layer_profiles],
        "activation_parameters_bytes": [p.activation_size for p in layer_profiles]
    } 