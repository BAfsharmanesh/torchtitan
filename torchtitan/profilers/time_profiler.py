from typing import Dict, List, Optional, Any
import contextlib
import time
import torch

from .base_profiler import BaseProfiler
from .hooks import register_timing_hooks


class TimeProfiler(BaseProfiler):
    """Profiles execution time of model layers during training.

    Tracks forward and backward pass timings for specified layers using PyTorch hooks.
    Can also record custom timing spans using context managers.
    """

    def __init__(self, layer_names: Optional[List[str]] = None):
        """Initialize time profiler

        Args:
            layer_names: List of layer names to profile. If None, profiles all layers.
        """
        super().__init__(layer_names)

        self.timings: Dict[str, List[float]] = {}
        self.memory_usage: Dict[str, List[int]] = {}
        self.hooks: Dict[str, Any] = {}

    def register_timing_hooks(
        self, model: torch.nn.Module, func: Optional[callable] = None
    ) -> None:
        """Register timing hooks on model layers

        Args:
            model: PyTorch model to profile
            func: Optional callback function to execute at hook points
        """
        register_timing_hooks(
            model, self.timings, self.memory_usage, self.layer_names, func, self.hooks
        )

    def get_timings(self) -> Dict[str, List[float]]:
        """Get raw timing measurements

        Returns:
            Dictionary mapping timing keys to lists of timestamps
        """
        return self.timings

    def get_memory_usage(self) -> Dict[str, List[int]]:
        """Get memory usage measurements

        Returns:
            Dictionary mapping layer names to lists of memory usage values
        """
        return self.memory_usage

    def reset_metrics(self):
        """Reset all timing measurements"""
        self.timings = {}

    def reset_memory_usage(self):
        """Reset memory usage tracking"""
        self.memory_usage = {}

    def get_duration_timings(self) -> Dict[str, List[float]]:
        """Calculate duration between start/end timing pairs

        Returns:
            Dictionary mapping layer names to lists of durations in milliseconds
        """
        timings = self.get_timings()
        duration_timings = {}
        for key, value in timings.items():
            if key.endswith("end"):
                start_key = key.replace("end", "start")
                duration_timings[key.replace("_end", "")] = [
                    (value[i] - timings[start_key][i]) * 1000 for i in range(len(value))
                ]
        return duration_timings

    def get_average_metrics(
        self, warm: int, active: int, layers_name: List[str]
    ) -> Dict[str, List[float]]:
        """Calculate average timing metrics over warm-up and active steps

        Args:
            warm: Number of warm-up steps to skip
            active: Number of active steps to average over
            layers_name: List of layer names to get metrics for

        Returns:
            Dictionary containing averaged timing metrics

        Raises:
            AssertionError: If active steps <= 0 or not enough samples
        """
        duration_timings = self.get_duration_timings()
        avg_timings = self._calculate_average_timings(duration_timings, warm, active)
        # avg_timings = {}
        # for key, value in duration_timings.items():
        #     assert (
        #         len(value) >= warm + active
        #     ), f"Number of timings for {key} is less than active+warm steps"
        #     avg_timings[key] = sum(value[warm : warm + active]) / (active)

        layer_totals = self._calculate_layer_totals(avg_timings, self.layer_names)

        # Validate and extract requested layers
        recorded_layer_names = [i[0] for i in layer_totals]
        self._validate_layer_names(recorded_layer_names, layers_name)

        avg_timings["layer_compute_total_ms"] = [
            layer_totals[recorded_layer_names.index(ln)][1] for ln in layers_name
        ]

        return avg_timings

    def _calculate_average_timings(
        self, duration_timings: Dict[str, List[float]], warm: int, active: int
    ) -> Dict[str, float]:
        """Average timing values over the active period"""
        avg_timings = {}
        for key, value in duration_timings.items():
            assert (
                len(value) >= warm + active
            ), f"Number of timings for {key} is less than active+warm steps"
            avg_timings[key] = sum(value[warm : warm + active]) / (active)
        return avg_timings

    def _calculate_layer_totals(
        self, avg_timings: Dict[str, float], layer_names: List[str]
    ) -> List[tuple[str, float]]:
        """Calculate total compute time per layer"""       
        layer_totals = {}
        for layer, value in avg_timings.items():
            layer_name = self._return_layer_name(layer)
            if layer_name not in layer_totals:
                layer_totals[layer_name] = 0
            if layer_name in layer_names:
                layer_totals[layer_name] += value

        return list(layer_totals.items())

    def _return_layer_name(self, name: str) -> str:
        """Extract base layer name by removing forward/backward suffixes"""
        return name.removesuffix("_backward").removesuffix("_forward")

    @contextlib.contextmanager
    def record_time(self, key: str, sync: bool = False):
        """Context manager to record execution time of a code block

        Args:
            key: Identifier for the timing measurement
            sync: Whether to synchronize CUDA operations
        """
        if key + "_start" not in self.timings:
            self.timings[key + "_start"] = []
            self.timings[key + "_end"] = []
        if sync:
            torch.cuda.synchronize()
        self.timings[key + "_start"].append(time.time())
        yield  # Yield control back to the calling context
        if sync:
            torch.cuda.synchronize()
        self.timings[key + "_end"].append(time.time())

    def record_time_tic(self, key: str, sync: bool = False) -> None:
        """Record start time for a measurement

        Args:
            key: Identifier for the timing measurement
            sync: Whether to synchronize CUDA operations
        """
        if key + "_start" not in self.timings:
            self.timings[key + "_start"] = []
            self.timings[key + "_end"] = []
        if sync:
            torch.cuda.synchronize()
        self.timings[key + "_start"].append(time.time())

    def record_time_toc(self, key: str, sync: bool = False) -> None:
        """Record end time for a measurement

        Args:
            key: Identifier for the timing measurement
            sync: Whether to synchronize CUDA operations

        Raises:
            AssertionError: If no matching start time exists
        """
        assert key + "_end" in self.timings, f"Key {key}_end not found in timings"
        if sync:
            torch.cuda.synchronize()
        self.timings[key + "_end"].append(time.time())
