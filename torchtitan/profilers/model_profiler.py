from typing import List, Dict, Tuple, Any, Optional, NamedTuple
import torch
from .constants import ACTIVATION_SAFETY_FACTOR, TOTAL_SAFETY_FACTOR

class LayerInfo(NamedTuple):
    """Information about a model layer."""
    name: str
    parameter_size: int
    activation_size: Optional[int] = None

class ModelLayerProfile:
    """Profiles model architecture and layer characteristics."""
    
    def __init__(self, model: torch.nn.Module, layer_names: Optional[List[str]] = None):
        """Initialize model profiler.
        
        Args:
            model: PyTorch model to profile
            layer_names: Optional list of layer names to profile
        """
        self.model = model.to("meta")
        self.layer_names = layer_names or self._get_default_layer_names()
        
    def _get_default_layer_names(self) -> List[str]:
        """Get default list of layer names to profile."""
        return [name for name, _ in self.model.named_modules()]
        
    def _get_layer_info(self, include_activations: bool = False, 
                       dummy_input: Optional[torch.Tensor] = None) -> Dict[str, LayerInfo]:
        """Get parameter and activation info for all layers.
        
        Args:
            include_activations: Whether to collect activation sizes
            dummy_input: Required if include_activations is True
            
        Returns:
            Dictionary mapping layer names to their info
        """
        layer_info = {}
        
        # Get parameter sizes
        for name, layer in self.model.named_modules():
            if name in self.layer_names:
                param_size = sum(p.element_size() * p.numel() for p in layer.parameters())
                layer_info[name] = LayerInfo(name=name, parameter_size=param_size)
                
        # Get activation sizes if requested
        if include_activations:
            if dummy_input is None:
                raise ValueError("dummy_input required to profile activations")
                
            activation_sizes = self._profile_activations(dummy_input)
            
            # Update existing LayerInfo objects with activation sizes
            for name, act_size in activation_sizes.items():
                if name in layer_info:
                    info = layer_info[name]
                    layer_info[name] = LayerInfo(
                        name=info.name,
                        parameter_size=info.parameter_size,
                        activation_size=act_size
                    )
                    
        return layer_info
        
    def _profile_activations(self, dummy_input: torch.Tensor) -> Dict[str, int]:
        """Profile activation sizes for all layers."""
        dummy_input = dummy_input.clone().to("meta")
        activation_sizes = {}
        hooks = []

        def hook_fn(name: str):
            def _hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
                if output is not None:
                    if isinstance(output, (tuple, list)):
                        size = sum(t.element_size() * t.numel() 
                                 for t in output if isinstance(t, torch.Tensor))
                    else:
                        size = output.element_size() * output.numel()
                    activation_sizes[name] = activation_sizes.get(name, 0) + size
            return _hook

        try:
            # Register hooks
            for name, module in self.model.named_modules():
                if name in self.layer_names:
                    hooks.append(module.register_forward_hook(hook_fn(name)))

            # Forward pass
            with torch.no_grad():
                self.model(dummy_input)
                
        finally:
            for hook in hooks:
                hook.remove()
                
        return activation_sizes

    def get_layer_profiles(self, dummy_input: Optional[torch.Tensor] = None) -> List[LayerInfo]:
        """Get profiling information for all requested layers.
        
        Args:
            dummy_input: Optional tensor for activation profiling
            
        Returns:
            List of LayerInfo objects in layer_names order
        """
        include_activations = dummy_input is not None
        layer_info = self._get_layer_info(include_activations, dummy_input)
        
        # Return info in requested order
        return [layer_info[name] for name in self.layer_names if name in layer_info]

def slice_layers_2_fit_gpu(
    act_weight_profiled: Dict[str, Any], 
    gpu_memory: float, 
    tp_degree: int
) -> List[List[int]]:
    """Slice model layers to fit within GPU memory constraints.
    
    Args:
        act_weight_profiled: Dictionary containing layer profiles
        gpu_memory: Available GPU memory in GB
        tp_degree: Tensor parallel degree
        
    Returns:
        List of layer index groups that fit in memory
    """
    parameters_per_layer = act_weight_profiled["parameters_per_layer_bytes"]
    activation_parameters = act_weight_profiled["activation_parameters_bytes"]
    model_name = "_".join(act_weight_profiled["model_name"].split("_")[:-1])

    assert len(parameters_per_layer) == len(activation_parameters), "Number of layers mismatch"
    
    def _predict_memory_usage(weight: int, act: int, model_name: str) -> float:
        """Predict memory usage for a layer."""
        return (TOTAL_SAFETY_FACTOR[model_name] * 
                (ACTIVATION_SAFETY_FACTOR[model_name] * act + weight * 4) / tp_degree)

    # Calculate per-layer sizes
    layer_sizes = [
        _predict_memory_usage(params, acts, model_name) / (1024 ** 3)
        for params, acts in zip(parameters_per_layer, activation_parameters)
    ]

    # Group layers into slices that fit in memory
    slices = []
    current_slice = []
    current_params = 0
    current_acts = 0

    for i, (params, acts) in enumerate(zip(parameters_per_layer, activation_parameters)):
        current_params += params
        current_acts += acts
        
        if _predict_memory_usage(current_params, current_acts, model_name) < gpu_memory:
            current_slice.append(i)
        else:
            if not current_slice:
                raise ValueError("Some layers are too big to fit in memory")
            slices.append(current_slice)
            current_slice = [i]
            current_params = params
            current_acts = acts

    if current_slice:
        slices.append(current_slice)

    return slices 