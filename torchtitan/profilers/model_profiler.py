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

        def hook_fn(parent_name: str):
            def _hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
                if output is not None:
                    if isinstance(output, (tuple, list)):
                        size = sum(t.element_size() * t.numel() 
                                 for t in output if isinstance(t, torch.Tensor))
                    else:
                        size = output.element_size() * output.numel()
                    activation_sizes[parent_name] = activation_sizes.get(parent_name, 0) + size
            return _hook

        try:
            # Register hooks for all submodules within each layer
            for name, layer in self.model.named_modules():
                if name in self.layer_names:
                    if len(list(layer.children())) > 0:
                        # For parent layers, hook all children
                        for submodule in layer.modules():
                            hooks.append(submodule.register_forward_hook(hook_fn(name)))
                    else:
                        # For leaf modules, just hook the module itself
                        hooks.append(layer.register_forward_hook(hook_fn(name)))

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
