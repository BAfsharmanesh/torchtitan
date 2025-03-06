import contextlib
import time
from typing import List

import torch

from .base_profiler import BaseProfiler
from .hooks import register_timing_hooks

class TimeProfiler(BaseProfiler):
    def __init__(self, layer_names: List[str] = None):
        """_summary_

        Args:
            layer_names (List[str], optional): list of layer names to profile. Defaults to None.
        """
        super().__init__(layer_names)

        self.timings = {}
        self.memory_usage = {}
        self.hooks = {}

    def register_timing_hooks(self, model, func=None):
        register_timing_hooks(
            model, self.timings, self.memory_usage, self.layer_names, func, self.hooks
        )

    def get_timings(self):
        return self.timings

    def get_memory_usage(self):
        return self.memory_usage

    def reset_metrics(self):
        self.timings = {}

    def reset_memory_usage(self):
        self.memory_usage = {}

    def get_duration_timings(self):
        timings = self.get_timings()
        duration_timings = {}
        for key, value in timings.items():
            if key.endswith("end"):
                start_key = key.replace("end", "start")
                duration_timings[key.replace("_end", "")] = [
                    (value[i] - timings[start_key][i]) * 1000 for i in range(len(value))
                ]
        return duration_timings

    def get_average_metrics(self, warm, active, layers_name):

        duration_timings = self.get_duration_timings()
        avg_timings = {}
        for key, value in duration_timings.items():
            assert (
                len(value) >= warm + active
            ), f"Number of timings for {key} is less than active+warm steps"
            avg_timings[key] = sum(value[warm : warm + active]) / (active)

        layer_compute_total_ms_dict = {}
        for layer, value in avg_timings.items():
            layer_name = self._return_layer_name(layer)
            if layer_name not in layer_compute_total_ms_dict:
                layer_compute_total_ms_dict[layer_name] = 0
            if layer_name in self.layer_names:
                layer_compute_total_ms_dict[layer_name] += value

        layer_compute_total_ms_dict = list(layer_compute_total_ms_dict.items())

        # avg_timings["layer_compute_total_ms"] = [
        #     i[1] for i in sorted(layer_compute_total_ms_dict) if i[0] in layers_name
        # ]

        recorded_layer_names = [i[0] for i in layer_compute_total_ms_dict]
        avg_timings["layer_compute_total_ms"] = []
        
        self._validate_layer_names(recorded_layer_names, layers_name)
        
        for ln in layers_name:
            avg_timings["layer_compute_total_ms"].append(
                layer_compute_total_ms_dict[recorded_layer_names.index(ln)][1]
            )

        return avg_timings

    def _return_layer_name(self, name):
        return name.removesuffix("_backward").removesuffix("_forward")

    @contextlib.contextmanager
    def record_time(self, key, sync):
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

    def record_time_tic(self, key, sync):
        if key + "_start" not in self.timings:
            self.timings[key + "_start"] = []
            self.timings[key + "_end"] = []
        if sync:
            torch.cuda.synchronize()
        self.timings[key + "_start"].append(time.time())

    def record_time_toc(self, key, sync):
        assert key + "_end" in self.timings, f"Key {key}_end not found in timings"
        if sync:
            torch.cuda.synchronize()
        self.timings[key + "_end"].append(time.time())