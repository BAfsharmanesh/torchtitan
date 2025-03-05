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
    return [name for name, _ in model.named_modules()]

def get_param_act_info(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    layer_names: List[str]
) -> Dict[str, Any]:
    """Get parameter and activation information for model layers.
    
    Args:
        model: PyTorch model to inspect
        dummy_input: Sample input tensor
        layer_names: List of layer names to analyze
        
    Returns:
        Dictionary containing layer parameter and activation sizes
    """
    # Get parameter sizes
    parameters_per_layer_bytes = []
    for name in layer_names:
        module = dict(model.named_modules())[name]
        param_size = sum(p.nelement() * p.element_size() for p in module.parameters())
        parameters_per_layer_bytes.append(param_size)
        
    # Get activation sizes using meta device
    model_profiler = ModelLayerProfile(model, layer_names)
    activation_info = model_profiler.get_activation_parameters_per_layer(dummy_input)
    activation_parameters_bytes = [size for _, size in activation_info]
    
    return {
        "model_name": model.__class__.__name__,
        "number_of_layers": len(layer_names),
        "parameters_per_layer_bytes": parameters_per_layer_bytes,
        "activation_parameters_bytes": activation_parameters_bytes,
        "total_parameters_bytes": sum(parameters_per_layer_bytes)
    } 