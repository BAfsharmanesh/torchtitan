import contextlib
import time
from typing import Dict, List, Any, Optional, Callable

import torch
from .base_profiler import BaseProfiler

class TimeProfiler(BaseProfiler):
    """Profiles execution time of model layers during forward and backward passes."""
    
    def __init__(self, layer_names: Optional[List[str]] = None):
        """Initialize the time profiler.
        
        Args:
            layer_names: List of layer names to profile. If None, all layers will be profiled.
        """
        super().__init__(layer_names)
        self.timings: Dict[str, List[float]] = {}
        
    def register_hooks(self, model: torch.nn.Module, callback: Optional[Callable] = None) -> None:
        """Register timing hooks on model layers.
        
        Args:
            model: PyTorch model to profile
            callback: Optional callback function to execute after timing
        """
        def start_time(layer_name: str, pass_type: str) -> Callable:
            def hook(module: torch.nn.Module, input: Any) -> None:
                torch.cuda.synchronize()
                self.timings.setdefault(f"{layer_name}_{pass_type}_start", []).append(time.time())
                if callback:
                    callback()
            return hook

        def end_time(layer_name: str, pass_type: str) -> Callable:
            def hook(module: torch.nn.Module, input: Any, output: Any) -> None:
                torch.cuda.synchronize()
                self.timings.setdefault(f"{layer_name}_{pass_type}_end", []).append(time.time())
                if callback:
                    callback()
            return hook

        # Register hooks only for specified layers
        for name, layer in model.named_modules():
            if name in self.layer_names:
                if name in self._hooks:
                    self.remove_hooks()
                
                hooks = [
                    layer.register_forward_pre_hook(start_time(name, "forward")),
                    layer.register_forward_hook(end_time(name, "forward")),
                    layer.register_full_backward_pre_hook(start_time(name, "backward")),
                    layer.register_full_backward_hook(end_time(name, "backward"))
                ]
                self._hooks[name] = hooks

    def reset(self) -> None:
        """Reset all timing measurements."""
        self.timings.clear()

    def get_metrics(self) -> Dict[str, List[float]]:
        """Get raw timing measurements.
        
        Returns:
            Dictionary mapping layer names to lists of timing measurements
        """
        return dict(self.timings)

    def get_duration_timings(self) -> Dict[str, List[float]]:
        """Calculate duration between start and end timings.
        
        Returns:
            Dictionary mapping layer names to lists of durations in milliseconds
        """
        duration_timings = {}
        for key, value in self.timings.items():
            if key.endswith("end"):
                start_key = key.replace("end", "start")
                duration_timings[key.replace("_end", "")] = [
                    (value[i] - self.timings[start_key][i]) * 1000 
                    for i in range(len(value))
                ]
        return duration_timings

    @contextlib.contextmanager
    def record_time(self, key: str, sync: bool = True):
        """Context manager for timing arbitrary code blocks.
        
        Args:
            key: Name for this timing measurement
            sync: Whether to synchronize CUDA operations
        """
        if key + "_start" not in self.timings:
            self.timings[key + "_start"] = []
            self.timings[key + "_end"] = []
        if sync:
            torch.cuda.synchronize()
        self.timings[key + "_start"].append(time.time())
        yield
        if sync:
            torch.cuda.synchronize()
        self.timings[key + "_end"].append(time.time())

    def record_time_tic(self, key: str, sync: bool = True):
        """Start timing measurement.
        
        Args:
            key: Name for this timing measurement
            sync: Whether to synchronize CUDA operations
        """
        if key + "_start" not in self.timings:
            self.timings[key + "_start"] = []
            self.timings[key + "_end"] = []
        if sync:
            torch.cuda.synchronize()
        self.timings[key + "_start"].append(time.time())

    def record_time_toc(self, key: str, sync: bool = True):
        """End timing measurement.
        
        Args:
            key: Name for this timing measurement
            sync: Whether to synchronize CUDA operations
        """
        assert key + "_end" in self.timings, f"No matching start time found for {key}"
        if sync:
            torch.cuda.synchronize()
        self.timings[key + "_end"].append(time.time()) 