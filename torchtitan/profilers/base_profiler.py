from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch

class BaseProfiler(ABC):
    """Base class for all profilers in TorchTitan.
    
    Provides common functionality and interfaces that all profilers should implement.
    """
    
    def __init__(self, layer_names: Optional[List[str]] = None):
        """Initialize the base profiler.
        
        Args:
            layer_names: List of layer names to profile. If None, all layers will be profiled.
        """
        self.layer_names = layer_names
        self._metrics: Dict[str, List[float]] = {}
        self._hooks: Dict[str, List[Any]] = {}


    @abstractmethod
    def reset(self) -> None:
        """Reset all collected metrics."""
        self._metrics.clear()

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get the raw collected metrics.
        
        Returns:
            Dictionary containing the collected metrics
        """
        return dict(self._metrics)

    def get_average_metrics(self, warm_steps: int, active_steps: int, layers_name: List[str]) -> Dict[str, Any]:
        """Calculate average metrics across warmup and active steps.
        
        Args:
            warm_steps: Number of warmup steps to skip
            active_steps: Number of active steps to average over
            layers_name: List of layer names to include in results
            
        Returns:
            Dictionary containing averaged metrics
        """
        assert active_steps > 0, "Active steps should be greater than 0"
        metrics = self.get_metrics()
        
        avg_metrics = {}
        for key, values in metrics.items():
            assert len(values) >= warm_steps + active_steps, \
                f"Not enough samples for {key}: need {warm_steps + active_steps}, got {len(values)}"
            avg_metrics[key] = sum(values[warm_steps:warm_steps + active_steps]) / active_steps
            
        return avg_metrics

    @staticmethod
    def _clean_layer_name(name: str) -> str:
        """Clean up layer name by removing common suffixes.
        
        Args:
            name: Raw layer name
            
        Returns:
            Cleaned layer name
        """
        return name.removesuffix("_backward").removesuffix("_forward")

    def _validate_layer_names(self, model: torch.nn.Module) -> None:
        """Validate that all specified layer names exist in the model.
        
        Args:
            model: PyTorch model to validate against
            
        Raises:
            ValueError: If any layer name is not found in the model
        """
        if not self.layer_names:
            return
            
        model_layers = dict(model.named_modules())
        for name in self.layer_names:
            if name not in model_layers:
                raise ValueError(f"Layer {name} not found in model. Available layers: {list(model_layers.keys())}") 