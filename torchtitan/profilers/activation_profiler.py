from typing import Dict, List, Optional, Any, Set
import torch
import contextlib

class SavedActivationContext:
    """Context manager for tracking activation memory during model execution."""
    
    def __init__(self, layer_names: Optional[List[str]] = None):
        """Initialize activation context.
        
        Args:
            layer_names: Optional list of layer names to track
        """
        self.layer_names = set(layer_names) if layer_names else None
        self.saved_tensors: Dict[str, List[torch.Tensor]] = {}
        self.hooks: List[Any] = []
        
    def _save_activation(self, name: str, tensor: torch.Tensor) -> None:
        """Save activation tensor for a layer."""
        if self.layer_names is None or name in self.layer_names:
            if name not in self.saved_tensors:
                self.saved_tensors[name] = []
            self.saved_tensors[name].append(tensor)
    
    def register_hooks(self, model: torch.nn.Module) -> None:
        """Register forward hooks on model layers.
        
        Args:
            model: PyTorch model to profile
        """
        def hook_fn(name: str):
            def _hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
                if isinstance(output, (tuple, list)):
                    for t in output:
                        if isinstance(t, torch.Tensor):
                            self._save_activation(name, t)
                elif isinstance(output, torch.Tensor):
                    self._save_activation(name, output)
            return _hook
            
        for name, module in model.named_modules():
            if self.layer_names is None or name in self.layer_names:
                self.hooks.append(module.register_forward_hook(hook_fn(name)))
                
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def get_activation_sizes(self) -> Dict[str, int]:
        """Get memory sizes of saved activations.
        
        Returns:
            Dictionary mapping layer names to activation sizes in bytes
        """
        sizes = {}
        for name, tensors in self.saved_tensors.items():
            sizes[name] = sum(t.nelement() * t.element_size() for t in tensors)
        return sizes
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        self.saved_tensors.clear()

@contextlib.contextmanager
def measure_activation_shape(model: torch.nn.Module, layer_names: Optional[List[str]] = None):
    """Context manager for measuring activation shapes during model execution.
    
    Args:
        model: PyTorch model to profile
        layer_names: Optional list of layer names to track
        
    Yields:
        SavedActivationContext instance
    """
    context = SavedActivationContext(layer_names)
    context.register_hooks(model)
    try:
        yield context
    finally:
        context.remove_hooks() 