import contextlib
import time
from typing import List

import torch

from torchtitan.profilers.base_profiler import BaseProfiler


class TimeProfiler(BaseProfiler):
    def __init__(self, layer_names: List[str] = None):
        """_summary_

        Args:
            layer_names (List[str], optional): list of layer names to profile. Defaults to None.
        """
        super().__init__(layer_names)
        # self.hook_layers = layer_names

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

    def reset_timings(self):
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

    def get_average_timings(self, warm, active, layers_name):
        assert active > 0, "Active steps should be greater than 0"

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
        for ln in layers_name:
            assert (
                ln in recorded_layer_names
            ), f"Layer {ln} not found in the model layers {recorded_layer_names}"
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


def register_timing_hooks(
    model, timings, memory_usage, hook_layers, func=None, hooks=None
):

    def start_time(layer_name, pass_type, func=None):
        def hook(module, input):
            torch.cuda.synchronize()  # Ensure all CUDA operations are finished
            timings.setdefault(f"{layer_name}_{pass_type}_start", []).append(
                time.time()
            )
            torch.cuda.reset_peak_memory_stats()
            if func is not None:
                func()

        return hook

    def end_time(layer_name, pass_type, func=None):
        def hook(module, input, output):
            torch.cuda.synchronize()  # Ensure all CUDA operations are finished
            timings.setdefault(f"{layer_name}_{pass_type}_end", []).append(time.time())

            memory_usage.setdefault(
                f"{layer_name}_{pass_type}_start_reserved", []
            ).append(torch.cuda.max_memory_reserved())

            memory_usage.setdefault(
                f"{layer_name}_{pass_type}_start_allocated", []
            ).append(torch.cuda.max_memory_allocated())

            if func is not None:
                func()

        return hook

    # Iterate over each layer and register hooks
    # Only apply hooks to container-like layers, not leaf layers
    # hook_layers = [f"layers.{i}" for i in range(10)]
    # hook_layers = hook_layers + ["norm", "output"]

    for name, layer in model.named_modules():
        if name in hook_layers:
            if name in hooks:
                for layer_hooks in hooks[name]:
                    layer_hooks.remove()
                # delete key name from hooks
                del hooks[name]
            # print("Registering hooks for layer", name)
            h1 = layer.register_forward_pre_hook(start_time(name, "forward"))
            h2 = layer.register_forward_hook(end_time(name, "forward", func))
            h3 = layer.register_full_backward_pre_hook(start_time(name, "backward"))
            h4 = layer.register_full_backward_hook(end_time(name, "backward"))
            hooks[name] = [h1, h2, h3, h4]
        # elif name in ["tok_embeddings"]:
        #     if name in hooks:
        #         for layer_hooks in hooks[name]:
        #             layer_hooks.remove()
        #         del hooks[name]
        #     h1 = layer.register_forward_pre_hook(start_time(name, "forward"))
        #     h2 = layer.register_forward_hook(end_time(name, "forward", func))
        #     hooks[name] = [h1, h2]