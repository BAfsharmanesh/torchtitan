from typing import List, Dict, Tuple, Any, Optional
import torch
from .constants import ACTIVATION_SAFETY_FACTOR, TOTAL_SAFETY_FACTOR

class ModelLayerProfile:
    """Profiles model architecture and layer characteristics."""
    
    def __init__(self, model: torch.nn.Module, layer_names: Optional[List[str]] = None):
        """Initialize model profiler.
        
        Args:
            model: PyTorch model to profile
            layer_names: Optional list of layer names to profile
        """
        self.model = model
        self.layer_names = layer_names or []
        self._activation_sizes: List[Tuple[str, int]] = []
        
    def get_activation_parameters_per_layer(self, dummy_input: torch.Tensor) -> List[Tuple[str, int]]:
        """Calculate activation sizes for each layer during forward pass.
        
        Args:
            dummy_input: Sample input tensor
            
        Returns:
            List of (layer_name, activation_size) tuples
        """
        dummy_input = dummy_input.clone().to("meta")
        hooks = []
        activation_parameters_bytes = []

        def hook_fn(name: str):
            def _hook(module: torch.nn.Module, input: Any, output: Any) -> None:
                if isinstance(output, (tuple, list)):
                    total_size = sum(t.nelement() * t.element_size() for t in output if isinstance(t, torch.Tensor))
                else:
                    total_size = output.nelement() * output.element_size()
                activation_parameters_bytes.append((name, total_size))
            return _hook

        # Register hooks for all named modules
        for name, module in self.model.named_modules():
            if not self.layer_names or name in self.layer_names:
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Forward pass to trigger hooks
        try:
            self.model(dummy_input)
        finally:
            for hook in hooks:
                hook.remove()

        # Aggregate activation sizes
        activation_dict = {}
        for name, size in activation_parameters_bytes:
            activation_dict[name] = activation_dict.get(name, 0) + size

        return list(activation_dict.items())

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