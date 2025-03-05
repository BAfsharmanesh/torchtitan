import contextlib
import time
from typing import Dict, List, Any, Optional, Callable

import torch
from .base_profiler import BaseProfiler

class TimeProfiler(BaseProfiler):
    """Profiles execution time of model layers."""
    
    def __init__(self, layer_names: Optional[List[str]] = None):
        """Initialize the time profiler.
        
        Args:
            layer_names: List of layer names to profile
        """
        super().__init__(layer_names)
        self.reset()
        
    def reset(self) -> None:
        """Reset all timing measurements."""
        self.layer_times = {ln: [] for ln in self.layer_names}
        self.total_forward_time = []
        self.total_backward_time = []
        
    def register_hooks(self, model: torch.nn.Module) -> None:
        """Register timing hooks on model layers."""
        for name, module in model.named_modules():
            if name in self.layer_names:
                forward_hook = module.register_forward_hook(self._forward_hook(name))
                backward_hook = module.register_backward_hook(self._backward_hook(name))
                self._hooks[name] = [forward_hook, backward_hook]
                
    def remove_hooks(self) -> None:
        """Remove all registered timing hooks."""
        super().remove_hooks()
        
    def _forward_hook(self, name: str):
        def hook(module, input, output):
            start_time = time.perf_counter()
            self.layer_times[name].append(("forward", start_time))
        return hook
        
    def _backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            end_time = time.perf_counter()
            self.layer_times[name].append(("backward", end_time))
        return hook
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get the timing metrics for all layers."""
        metrics = {}
        for name, times in self.layer_times.items():
            forward_times = [t[1] for t in times if t[0] == "forward"]
            backward_times = [t[1] for t in times if t[0] == "backward"]
            metrics[name] = {
                "forward_time": sum(forward_times) / len(forward_times) if forward_times else 0,
                "backward_time": sum(backward_times) / len(backward_times) if backward_times else 0
            }
        return metrics

    def get_duration_timings(self) -> Dict[str, List[float]]:
        """Calculate duration between start and end timings.
        
        Returns:
            Dictionary mapping layer names to lists of durations in milliseconds
        """
        duration_timings = {}
        for key, value in self.layer_times.items():
            forward_times = [t[1] for t in value if t[0] == "forward"]
            backward_times = [t[1] for t in value if t[0] == "backward"]
            if forward_times and backward_times:
                duration_timings[key] = [
                    (backward_times[i] - forward_times[i]) * 1000 for i in range(len(forward_times))
                ]
        return duration_timings

    @contextlib.contextmanager
    def record_time(self, key: str, sync: bool = True):
        """Context manager for timing arbitrary code blocks.
        
        Args:
            key: Name for this timing measurement
            sync: Whether to synchronize CUDA operations
        """
        if key + "_start" not in self.layer_times:
            self.layer_times[key + "_start"] = []
            self.layer_times[key + "_end"] = []
        if sync:
            torch.cuda.synchronize()
        self.layer_times[key + "_start"].append(time.time())
        yield
        if sync:
            torch.cuda.synchronize()
        self.layer_times[key + "_end"].append(time.time())

    def record_time_tic(self, key: str, sync: bool = True):
        """Start timing measurement.
        
        Args:
            key: Name for this timing measurement
            sync: Whether to synchronize CUDA operations
        """
        if key + "_start" not in self.layer_times:
            self.layer_times[key + "_start"] = []
            self.layer_times[key + "_end"] = []
        if sync:
            torch.cuda.synchronize()
        self.layer_times[key + "_start"].append(time.time())

    def record_time_toc(self, key: str, sync: bool = True):
        """End timing measurement.
        
        Args:
            key: Name for this timing measurement
            sync: Whether to synchronize CUDA operations
        """
        assert key + "_end" in self.layer_times, f"No matching start time found for {key}"
        if sync:
            torch.cuda.synchronize()
        self.layer_times[key + "_end"].append(time.time()) 