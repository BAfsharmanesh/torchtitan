from typing import List

import torch

from .base_profiler import BaseProfiler

class MemoryProfiler(BaseProfiler):
    def __init__(self, layer_names: List[str] = None):
        """_summary_

        Args:
            layer_names (List[str], optional): layer_names should be sorted in the order of forward pass. Defaults to None.
        """
        super().__init__(layer_names)

        self.reset_metrics()

    def log_activation_memory_info(self, saved_tensor_mem_layer: list[float]):
        assert len(saved_tensor_mem_layer) == len(
            self.layer_names
        ), "Number of layers and memory usage list should match"
        for layer_mem, ln in zip(saved_tensor_mem_layer, self.layer_names):
            self.activation_memory_usage[ln].append(layer_mem)

        self.total_activation_mem_size.append(sum(saved_tensor_mem_layer))

    def log_weight_grad_optimizer_memory_info(self, model, optimizers, device):
        """ log weight, grad, and optimizer memory usage for each layer in self.layer_names

        Args:
            model (_type_): _description_
            optimizers (_type_): _description_
            device (_type_): _description_

        Raises:
            ValueError: _description_
        """

        total_weight_size = 0
        total_grad_size = 0
        
        # iterate over each layer and get the memort address of the parameters, weights and grads memory usage
        # if the layer is in self.layer_names
        id_layer_param_num = {}
        id_param_num_untracked = {}
        for name, layer in model.named_modules():
            if name in self.layer_names:
                
                # get memory address of the parameters of a layer
                for t in layer.parameters():
                    assert t.device == device
                    if isinstance(t, torch.distributed.tensor.DTensor):
                        storage_id = id(t.to_local().untyped_storage())
                    else:
                        storage_id = id(t.untyped_storage())
                    id_layer_param_num[storage_id] = {"layer": name}

                # get memory usage of the parameters for a layer
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

                # get memory usage of the grads for a layer
                grad_size_layer = (
                    sum(
                        [
                            (
                                t.grad.to_local().untyped_storage().nbytes()
                                if isinstance(t.grad, torch.distributed.tensor.DTensor)
                                else t.grad.untyped_storage().nbytes()
                            )
                            for t in layer.parameters()
                            if t.grad is not None
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
            
            else:
                # get memory address of the parameters of a layer
                for t in layer.parameters():
                    assert t.device == device
                    if isinstance(t, torch.distributed.tensor.DTensor):
                        storage_id = id(t.to_local().untyped_storage())
                    else:
                        storage_id = id(t.untyped_storage())
                    id_param_num_untracked[storage_id] = {"layer": name}

        # print("total weight size:", total_weight_size, "MB")
        # print("total grads size:", total_grad_size, "MB")
        # print("total optimizer state size:", total_weight_size * 2, "MB")

        self.total_weight_mem_size.append(total_weight_size)
        self.total_grad_mem_size.append(total_grad_size)

        # assign param_num to each parameter in the model, so each parameter can be identified by its index and storage id
        for t_n, t in enumerate(model.parameters()):
            if isinstance(t, torch.distributed.tensor.DTensor):
                storage_id = id(t.to_local().untyped_storage())
            else:
                storage_id = id(t.untyped_storage())

            if storage_id not in id_layer_param_num:
                # it must be in id_param_num_untracked
                if storage_id not in id_param_num_untracked:
                    raise ValueError(
                        f"Layer not found for the parameter with storage id {storage_id}"
                    )
            else:
                id_layer_param_num[storage_id]["param_num"] = t_n

        # print("id_layer_paramnum:", id_layer_paramnum)

        # dict with key: param_num, value: layer and id
        param_num_layer_id = {
            v["param_num"]: {"layer": v["layer"], "id": k}
            for k, v in id_layer_param_num.items()
        }

        # list of layers in the model that we recorded memory usage for
        layer_list = [v["layer"] for k, v in id_layer_param_num.items()]

        # print(optimizers.optimizers[0].state_dict())
        state = optimizers.state_dict()["state"]
        # params = optimizers.optimizers[0].state_dict()["param_groups"][0]["params"]
        
        # iterate over each layer and get the memory address of the optimizer state for each layer
        # if the layer is in (layer_list) self.layer_names
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
                "activation": self.total_activation_mem_size,
                "total_memory": [i * 1024 for i in self.max_reserved_gib],
            },
        }

    def get_average_metrics(self, warm, active, layers_name):

        avg_mem_usage = {}
        total_res = self.get_memory_usage()
        for key, value in total_res.items():
            avg_mem_usage[key] = {}
            for ln, mem in value.items():
                assert (
                    len(mem) >= warm + active
                ), f"Number of memory usage for {ln} is less than active+warm steps"
                avg_mem_usage[key][ln] = sum(mem[warm : warm + active]) / (active)

        self.layer_memory_total_mb = []
        for ln in self.layer_names:
            self.layer_memory_total_mb.append(
                (
                    ln,
                    avg_mem_usage["activation"][ln]
                    + avg_mem_usage["weight"][ln]
                    + avg_mem_usage["grad"][ln]
                    + avg_mem_usage["optimizer"][ln],
                )
            )

        # self.layer_memory_total_mb = [
        #     i[1] for i in sorted(self.layer_memory_total_mb) if i[0] in layers_name
        # ]
        # avg_mem_usage["layer_memory_total_mb"] = self.layer_memory_total_mb

        recorded_layer_names = [i[0] for i in self.layer_memory_total_mb]
        avg_mem_usage["layer_memory_total_mb"] = []
        
        self._validate_layer_names(recorded_layer_names, layers_name)
        for ln in layers_name:
            avg_mem_usage["layer_memory_total_mb"].append(
                self.layer_memory_total_mb[recorded_layer_names.index(ln)][1]
            )

        return avg_mem_usage

    def reset_metrics(self):
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