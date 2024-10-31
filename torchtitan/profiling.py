# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import pickle
import time

import torch

from torchtitan.config_manager import JobConfig
from torchtitan.logging import logger

# the number of warmup steps before the active step in each profiling cycle
WARMUP = 3

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


@contextlib.contextmanager
def maybe_enable_profiling(config: JobConfig, *, global_step: int = 0):
    # get user defined profiler settings
    enable_profiling = config.profiling.enable_profiling

    if enable_profiling:
        dump_dir = config.job.dump_folder
        save_trace_dir = config.profiling.save_traces_folder
        trace_dir = os.path.join(dump_dir, save_trace_dir)
        profile_freq = config.profiling.profile_freq

        rank = torch.distributed.get_rank()

        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            logger.info(f"Dumping traces at step {prof.step_num}")
            begin = time.monotonic()
            prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.json")
            logger.info(
                f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds"
            )

        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        warmup, active = WARMUP, 1
        wait = profile_freq - (active + warmup)
        assert (
            wait >= 0
        ), "profile_freq must be greater than or equal to warmup + active"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


@contextlib.contextmanager
def maybe_enable_memory_snapshot(config: JobConfig, *, global_step: int = 0):
    enable_snapshot = config.profiling.enable_memory_snapshot
    if enable_snapshot:
        snapshot_folder = config.profiling.save_memory_snapshot_folder
        snapshot_dir = os.path.join(config.job.dump_folder, snapshot_folder)
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir, exist_ok=True)
        rank = torch.distributed.get_rank()

        class MemoryProfiler:
            def __init__(self, step_num: int, freq: int):
                torch.cuda.memory._record_memory_history(
                    max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES
                )
                # when resume training, we start from the last step
                self.step_num = step_num
                self.freq = freq

            def step(self, exit_ctx: bool = False):
                self.step_num += 1
                if not exit_ctx and self.step_num % self.freq != 0:
                    return
                if not exit_ctx:
                    curr_step = self.step_num
                    dir_name = f"iteration_{curr_step}"
                else:
                    # dump as iteration_0_exit if OOM at iter 1
                    curr_step = self.step_num - 1
                    dir_name = f"iteration_{curr_step}_exit"
                curr_snapshot_dir = os.path.join(snapshot_dir, dir_name)
                if not os.path.exists(curr_snapshot_dir):
                    os.makedirs(curr_snapshot_dir, exist_ok=True)
                logger.info(f"Dumping memory snapshot at step {curr_step}")
                begin = time.monotonic()
                with open(
                    f"{curr_snapshot_dir}/rank{rank}_memory_snapshot.pickle", "wb"
                ) as output:
                    pickle.dump(torch.cuda.memory._snapshot(), output)
                logger.info(
                    f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds"
                )

        logger.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
        profiler = MemoryProfiler(global_step, config.profiling.profile_freq)
        try:
            yield profiler
        except torch.OutOfMemoryError as e:
            profiler.step(exit_ctx=True)
    else:
        yield None


import time

import torch

#


def register_timing_hooks(model, timings, memory_usage, func=None):

    def start_time(layer_name, pass_type, func=None):
        def hook(module, input):
            torch.cuda.synchronize()  # Ensure all CUDA operations are finished
            timings[f"{layer_name}_{pass_type}_start"] = time.time()
            torch.cuda.reset_peak_memory_stats()
            if func is not None:
                func()

        return hook

    def end_time(layer_name, pass_type, func=None):
        def hook(module, input, output):
            torch.cuda.synchronize()  # Ensure all CUDA operations are finished
            timings[f"{layer_name}_{pass_type}_end"] = time.time()
            memory_usage[f"{layer_name}_{pass_type}_start_reserved"] = (
                torch.cuda.max_memory_reserved()
            )
            memory_usage[f"{layer_name}_{pass_type}_start_allocated"] = (
                torch.cuda.max_memory_allocated()
            )
            if func is not None:
                func()

        return hook

    # Iterate over each layer and register hooks
    for name, layer in model.named_modules():
        # Only apply hooks to container-like layers, not leaf layers
        hook_layers = [f"layers.{i}" for i in range(8)]
        hook_layers = hook_layers + ["norm", "output"]
        if name in hook_layers:
            print("Registering hooks for layer", name)
            layer.register_forward_pre_hook(start_time(name, "forward"))
            layer.register_forward_hook(end_time(name, "forward", func))
            layer.register_full_backward_pre_hook(start_time(name, "backward"))
            layer.register_full_backward_hook(end_time(name, "backward"))
        elif name in ["tok_embeddings"]:
            layer.register_forward_pre_hook(start_time(name, "forward"))
            layer.register_forward_hook(end_time(name, "forward"))


import weakref
from typing import Any, Iterable, Optional, Union

from torchtitan.utils import Color

color = Color


class SavedTensorContext:
    def __init__(
        self,
        ignored_tensors: Optional[Iterable[torch.Tensor]] = None,
    ) -> None:
        self._ignored_data_ptrs = (
            set()
            if ignored_tensors is None
            else {
                (
                    id(t.to_local().untyped_storage())
                    if isinstance(t, torch.distributed.tensor.DTensor)
                    else id(t.untyped_storage())
                )
                for t in ignored_tensors
            }
        )

        self.saved_tensor_dict = torch.utils.weak.WeakTensorKeyDictionary()
        self.saved_tensor_list = WeakTensorList()
        self.layer_pos = [
            0,
        ]

        def pack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            # str_saved = f"{Color.red}saved_tensor: {saved_tensor.device}, {saved_tensor.shape} {type(saved_tensor)}{Color.reset}"
            # str_local = f"{Color.red}local: {saved_tensor.to_local().device}, {saved_tensor.to_local().shape} {type(saved_tensor.to_local())}{Color.reset}"
            # logger.info(str_saved+str_local)
            # torch.cuda.synchronize()
            # logger.info(f"{color.red}storage: {type(saved_tensor)}{color.reset}")
            data_ptr = (
                id(saved_tensor.to_local().untyped_storage())
                if isinstance(saved_tensor, torch.distributed.tensor.DTensor)
                else id(saved_tensor.untyped_storage())
            )
            # logger.info(f"{color.red}storage: {type(data_ptr)}{color.reset}")
            if data_ptr not in self._ignored_data_ptrs:
                self.saved_tensor_dict[
                    (
                        saved_tensor.to_local()
                        if isinstance(saved_tensor, torch.distributed.tensor.DTensor)
                        else saved_tensor
                    )
                ] = data_ptr
                self.saved_tensor_list.append(
                    saved_tensor.to_local()
                    if isinstance(saved_tensor, torch.distributed.tensor.DTensor)
                    else saved_tensor
                )
            return saved_tensor

        def unpack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            return saved_tensor

        self._saved_tensors_hook = torch.autograd.graph.saved_tensors_hooks(
            pack_hook, unpack_hook
        )

    def take_layer_pos(self):
        # print("Taking layer pos", len(self.saved_tensor_list))
        self.layer_pos.append(len(self.saved_tensor_list))

    def __enter__(self) -> "SavedTensorContext":
        self._saved_tensors_hook.__enter__()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self._saved_tensors_hook.__exit__(*args, **kwargs)

    @property
    def saved_tensor_mem(self) -> int:
        """
        The memory in bytes of all saved tensors, accounting for views into the same storage.
        """
        accounted_for = self._ignored_data_ptrs.copy()
        total_bytes = 0
        for t in self.saved_tensor_dict:
            data_ptr = id(t.untyped_storage())
            if data_ptr not in accounted_for:
                # logger.info(f"{color.red}storage: {data_ptr}, size:, {t.untyped_storage().nbytes()/1024/1024} {t.shape}{color.reset}")
                # if t.untyped_storage().nbytes()/1024/1024 > 128:
                #     print(t.shape, t.untyped_storage().nbytes()/1024/1024, t.dtype, t.device)
                total_bytes += t.untyped_storage().nbytes()
                accounted_for.add(data_ptr)
        return total_bytes / 1024 / 1024

    @property
    def saved_tensor_mem_layer(self) -> list:
        """
        The memory in bytes of all saved tensors, accounting for views into the same storage.
        """
        accounted_for = self._ignored_data_ptrs.copy()
        total_bytes_list = []
        for layer_idx in range(len(self.layer_pos[:-1])):
            initial_idx = self.layer_pos[layer_idx]
            final_idx = self.layer_pos[layer_idx + 1]
            total_bytes = 0
            for i in range(initial_idx, final_idx):
                t = self.saved_tensor_list[i]
                if t is None:
                    continue
                data_ptr = id(t.untyped_storage())
                if data_ptr not in accounted_for:
                    total_bytes += t.untyped_storage().nbytes()
                    accounted_for.add(data_ptr)
            total_bytes_list.append(total_bytes / 1024 / 1024)
        return total_bytes_list


class WeakTensorList:
    def __init__(self):
        self._refs = []

    def append(self, tensor):
        # Add a weak reference to the tensor
        self._refs.append(weakref.ref(tensor))

    def __getitem__(self, index):
        # Retrieve the tensor, if it's still alive
        tensor_ref = self._refs[index]()
        # if tensor_ref is None:
        #     print(f"Tensor at index {index} has been garbage collected.")
        return tensor_ref

    def __len__(self):
        return len(self._refs)

    def cleanup(self):
        # Clean up any None references from the list
        self._refs = [ref for ref in self._refs if ref() is not None]
