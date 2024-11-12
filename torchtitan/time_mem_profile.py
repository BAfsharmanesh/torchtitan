import contextlib
import time
from typing import List

import torch


class LayerTimeProfiler:
    def __init__(self, layer_names: List[str] = None):
        """_summary_

        Args:
            layer_names (List[str], optional): list of layer names to profile. Defaults to None.
        """
        self.hook_layers = layer_names

        self.timings = {}
        self.memory_usage = {}
        self.hooks = {}

    def register_timing_hooks(self, model, func=None):
        register_timing_hooks(
            model, self.timings, self.memory_usage, self.hook_layers, func, self.hooks
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

    def get_average_timings(self, warm, active):
        assert active > 0, "Active steps should be greater than 0"

        duration_timings = self.get_duration_timings()
        avg_timings = {}
        for key, value in duration_timings.items():
            assert (
                len(value) >= warm + active
            ), f"Number of timings for {key} is less than active+warm steps"
            avg_timings[key] = sum(value[warm : warm + active]) / (active)
        return avg_timings

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


def get_layer_names(model):
    names = []
    for name, child in model.named_children():
        if name == "layers":
            for name_2, child_2 in child.named_children():
                names.append(name + "." + name_2)
            continue
        # print name of the layer
        names.append(name)
    return names


class LayerMemoryProfiler:
    def __init__(self, layer_names: List[str] = None):
        """_summary_

        Args:
            layer_names (List[str], optional): layer_names should be sorted in the order of forward pass. Defaults to None.
        """

        self.layer_names = layer_names

        self.reset_memory_usage()

        # self.activation_memory_usage = {}
        # self.weight_memory_usage = {}
        # self.grad_memory_usage = {}
        # self.optimizer_memory_usage = {}
        # for ln in self.layer_names:
        #     self.activation_memory_usage[ln] = []
        #     self.weight_memory_usage[ln] = []
        #     self.grad_memory_usage[ln] = []
        #     self.optimizer_memory_usage[ln] = []

        # self.total_weight_mem_size = []
        # self.total_grad_mem_size = []
        # self.total_optimizer_mem_size = []

    def log_activation_memory_info(self, saved_tensor_mem_layer: list[float]):
        assert len(saved_tensor_mem_layer) == len(
            self.layer_names
        ), "Number of layers and memory usage list should match"
        for layer_mem, ln in zip(saved_tensor_mem_layer, self.layer_names):
            self.activation_memory_usage[ln].append(layer_mem)
        
        self.total_activation_mem_size.append(sum(saved_tensor_mem_layer))

    def log_weight_grad_optimizer_memory_info(self, model, optimizers, device):

        total_weight_size = 0
        total_grad_size = 0
        # print weight size, grad size, and optimizer state size for each layers
        id_layer_param_num = {}
        for name, layer in model.named_modules():
            if name in self.layer_names:
                for t in layer.parameters():
                    assert t.device == device
                    if isinstance(t, torch.distributed.tensor.DTensor):
                        id_layer_param_num[id(t.to_local().untyped_storage())] = {
                            "layer": name
                        }
                    else:
                        id_layer_param_num[id(t.untyped_storage())] = {"layer": name}

                weight_size_layer = (
                    sum(
                        [
                            (
                                t.to_local().untyped_storage().nbytes()
                                if isinstance(t, torch.distributed.tensor.DTensor)
                                else t.untyped_storage().nbytes()
                            )
                            for t in layer.parameters()
                        ]
                    )
                    / 1024
                    / 1024
                )

                grad_size_layer = (
                    sum(
                        [
                            (
                                t.grad.to_local().untyped_storage().nbytes()
                                if isinstance(t.grad, torch.distributed.tensor.DTensor)
                                else t.grad.untyped_storage().nbytes()
                            )
                            for t in layer.parameters()
                        ]
                    )
                    / 1024
                    / 1024
                )
                # print(name, "weights", weight_size_layer, "MB", "grads", grad_size_layer, "MB")
                self.weight_memory_usage[name].append(weight_size_layer)
                self.grad_memory_usage[name].append(grad_size_layer)
                total_weight_size += weight_size_layer
                total_grad_size += grad_size_layer

        # print("total weight size:", total_weight_size, "MB")
        # print("total grads size:", total_grad_size, "MB")
        # print("total optimizer state size:", total_weight_size * 2, "MB")

        self.total_weight_mem_size.append(total_weight_size)
        self.total_grad_mem_size.append(total_grad_size)

        for t_n, t in enumerate(model.parameters()):
            if isinstance(t, torch.distributed.tensor.DTensor):
                id_layer_param_num[id(t.to_local().untyped_storage())][
                    "param_num"
                ] = t_n
            else:
                id_layer_param_num[id(t.untyped_storage())]["param_num"] = t_n

        # print("id_layer_paramnum:", id_layer_paramnum)

        param_num_layer_id = {
            v["param_num"]: {"layer": v["layer"], "id": k}
            for k, v in id_layer_param_num.items()
        }

        layer_list = [v["layer"] for k, v in id_layer_param_num.items()]

        # print(optimizers.optimizers[0].state_dict())
        state = optimizers.state_dict()["state"]
        # params = optimizers.optimizers[0].state_dict()["param_groups"][0]["params"]
        optimizer_mem = 0
        optimizer_mem_layer = {}
        for layer in layer_list:
            optimizer_mem_layer[layer] = 0
        for k in state.keys():
            for t in state[k].values():
                if isinstance(t, torch.distributed.tensor.DTensor):
                    optimizer_mem_layer[param_num_layer_id[k]["layer"]] += (
                        t.to_local().untyped_storage().nbytes() / 1024 / 1024
                    )
                    optimizer_mem += (
                        t.to_local().untyped_storage().nbytes() / 1024 / 1024
                    )
                else:
                    optimizer_mem_layer[param_num_layer_id[k]["layer"]] += (
                        t.untyped_storage().nbytes() / 1024 / 1024
                    )
                    optimizer_mem += t.untyped_storage().nbytes() / 1024 / 1024

        # print("total optimizer state size:", optimizer_mem, "MB")
        # print(
        #     "optimizer_mem_layer:",
        #     optimizer_mem_layer,
        #     sum(optimizer_mem_layer.values()),
        # )

        for ln, mem in optimizer_mem_layer.items():
            self.optimizer_memory_usage[ln].append(mem)
        self.total_optimizer_mem_size.append(optimizer_mem)

    def get_memory_usage(self):
        return {
            "activation": self.activation_memory_usage,
            "weight": self.weight_memory_usage,
            "grad": self.grad_memory_usage,
            "optimizer": self.optimizer_memory_usage,
            "total": {
                "weight": self.total_weight_mem_size,
                "grad": self.total_grad_mem_size,
                "optimizer": self.total_optimizer_mem_size,
                "max_reserved_gib": self.max_reserved_gib,
            },
        }

    def get_average_memory_usage(self, warm, active):
        assert active > 0, "Active steps should be greater than 0"

        avg_mem_usage = {}
        for key, value in self.get_memory_usage().items():
            avg_mem_usage[key] = {}
            for ln, mem in value.items():
                assert (
                    len(mem) >= warm + active
                ), f"Number of memory usage for {ln} is less than active+warm steps"
                avg_mem_usage[key][ln] = sum(mem[warm : warm + active]) / (active)

        return avg_mem_usage

    def reset_memory_usage(self):
        self.activation_memory_usage = {}
        self.weight_memory_usage = {}
        self.grad_memory_usage = {}
        self.optimizer_memory_usage = {}
        for ln in self.layer_names:
            self.activation_memory_usage[ln] = []
            self.weight_memory_usage[ln] = []
            self.grad_memory_usage[ln] = []
            self.optimizer_memory_usage[ln] = []

        self.total_activation_mem_size = []
        self.total_weight_mem_size = []
        self.total_grad_mem_size = []
        self.total_optimizer_mem_size = []
        self.max_reserved_gib = []

    def log_max_reserved_gib(self, max_reserved_gib):
        self.max_reserved_gib.append(max_reserved_gib)


import weakref
from typing import Any, Iterable, Optional, Union

from torchtitan.utils import Color

color = Color


class SavedActivationContext:
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

    def __enter__(self) -> "SavedActivationContext":
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
