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
        self.timings = {}  # Changed from layer_times to match original
        self.memory_usage = {}  # Added to match original
        self._hooks = {}
        
    def register_timing_hooks(self, model: torch.nn.Module, func: Optional[Callable] = None) -> None:
        """Register timing hooks on model layers (compatibility method)."""
        self.register_hooks(model)
        
    def register_hooks(self, model: torch.nn.Module) -> None:
        """Register timing hooks on model layers."""
        for name, module in model.named_modules():
            if name in self.layer_names:
                # Register start and end timing hooks
                forward_pre_hook = module.register_forward_pre_hook(self._forward_pre_hook(name))
                forward_hook = module.register_forward_hook(self._forward_hook(name))
                backward_hook = module.register_full_backward_hook(self._backward_hook(name))
                self._hooks[name] = [forward_pre_hook, forward_hook, backward_hook]
                
    def remove_hooks(self) -> None:
        """Remove all registered timing hooks."""
        super().remove_hooks()
        
    def _forward_pre_hook(self, name: str):
        def hook(module, input):
            if name not in self.timings:
                self.timings[f"{name}_start"] = []
                self.timings[f"{name}_end"] = []
            self.timings[f"{name}_start"].append(time.perf_counter())
        return hook
        
    def _forward_hook(self, name: str):
        def hook(module, input, output):
            self.timings[f"{name}_end"].append(time.perf_counter())
        return hook
        
    def _backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            if f"{name}_backward" not in self.timings:
                self.timings[f"{name}_backward"] = []
            self.timings[f"{name}_backward"].append(time.perf_counter())
        return hook
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get timing metrics for all layers."""
        metrics = {}
        for name in self.layer_names:
            if f"{name}_start" in self.timings and f"{name}_end" in self.timings:
                forward_times = [
                    (end - start) * 1000  # Convert to milliseconds
                    for start, end in zip(
                        self.timings[f"{name}_start"],
                        self.timings[f"{name}_end"]
                    )
                ]
                metrics[name] = {
                    "forward_time": sum(forward_times) / len(forward_times) if forward_times else 0
                }
        return metrics

    def get_duration_timings(self) -> Dict[str, List[float]]:
        """Calculate duration between start and end timings.
        
        Returns:
            Dictionary mapping layer names to lists of durations in milliseconds
        """
        duration_timings = {}
        for key, value in self.timings.items():
            if key.endswith("_start") and key.replace("_start", "_end") in self.timings:
                start_times = [t for t in value if t.endswith("_start")]
                end_times = [t for t in value if t.endswith("_end")]
                if start_times and end_times:
                    duration_timings[key[:-5]] = [
                        (end - start) * 1000 for start, end in zip(start_times, end_times)
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
        self.timings[key + "_start"].append(time.perf_counter())
        yield
        if sync:
            torch.cuda.synchronize()
        self.timings[key + "_end"].append(time.perf_counter())

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
        self.timings[key + "_start"].append(time.perf_counter())

    def record_time_toc(self, key: str, sync: bool = True):
        """End timing measurement.
        
        Args:
            key: Name for this timing measurement
            sync: Whether to synchronize CUDA operations
        """
        assert key + "_end" in self.timings, f"No matching start time found for {key}"
        if sync:
            torch.cuda.synchronize()
        self.timings[key + "_end"].append(time.perf_counter()) 