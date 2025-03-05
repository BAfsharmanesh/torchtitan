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
        self.hook_layers = layer_names  # Match original naming
        self.timings = {}
        self.memory_usage = {}
        self.hooks = {}
        
    def reset(self) -> None:
        """Reset all timing measurements."""
        self.timings.clear()
        self.memory_usage.clear()
        self.hooks.clear()
        
    def register_timing_hooks(self, model: torch.nn.Module, func: Optional[Callable] = None) -> None:
        """Register timing hooks on model layers."""
        for name, module in model.named_modules():
            if name in self.hook_layers:
                # Start timing
                forward_pre_hook = module.register_forward_pre_hook(self._forward_pre_hook(name))
                # End timing
                forward_hook = module.register_forward_hook(self._forward_hook(name))
                self.hooks[name] = [forward_pre_hook, forward_hook]
        
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hooks in self.hooks.values():
            for hook in hooks:
                hook.remove()
        self.hooks.clear()
        
    def _forward_pre_hook(self, name: str):
        def hook(module, input):
            if f"{name}_start" not in self.timings:
                self.timings[f"{name}_start"] = []
            self.timings[f"{name}_start"].append(time.perf_counter())
        return hook
        
    def _forward_hook(self, name: str):
        def hook(module, input, output):
            if f"{name}_end" not in self.timings:
                self.timings[f"{name}_end"] = []
            self.timings[f"{name}_end"].append(time.perf_counter())
        return hook
        
    def get_average_timings(self, warm: int, active: int, layers_name: List[str]) -> Dict[str, Any]:
        """Get average timings across steps.
        
        Args:
            warm: Number of warmup steps to skip
            active: Number of active steps to average over
            layers_name: List of layer names to include in results
            
        Returns:
            Dictionary containing averaged timings
        """
        assert active > 0, "Active steps should be greater than 0"
        duration_timings = self.get_duration_timings()
        avg_timings = {}
        
        for key, value in duration_timings.items():
            assert len(value) >= warm + active, \
                f"Number of timings for {key} is less than active+warm steps"
            avg_timings[key] = sum(value[warm:warm + active]) / active

        # Calculate layer compute times
        layer_compute_total_ms = []
        for name in layers_name:
            if f"{name}_start" in self.timings and f"{name}_end" in self.timings:
                times = []
                for start, end in zip(
                    self.timings[f"{name}_start"][warm:warm + active],
                    self.timings[f"{name}_end"][warm:warm + active]
                ):
                    times.append((end - start) * 1000)  # Convert to ms
                layer_compute_total_ms.append((name, sum(times) / len(times) if times else 0))

        # Add layer compute times to results
        avg_timings["layer_compute_total_ms"] = [t[1] for t in layer_compute_total_ms]

        return avg_timings

    def get_metrics(self) -> Dict[str, Any]:
        """Get timing metrics for all layers."""
        return self.get_average_timings(0, 0, self.hook_layers)

    def get_duration_timings(self) -> Dict[str, List[float]]:
        """Calculate duration between start and end timings."""
        duration_timings = {}
        for name in self.hook_layers:
            if f"{name}_start" in self.timings and f"{name}_end" in self.timings:
                durations = []
                for start, end in zip(self.timings[f"{name}_start"], 
                                    self.timings[f"{name}_end"]):
                    durations.append((end - start) * 1000)  # Convert to ms
                duration_timings[name] = durations
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