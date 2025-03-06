from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
from torch.optim import Optimizer

from .base_profiler import BaseProfiler


@dataclass
class TensorMemoryInfo:
    storage_id: int  # unique storage id of the tensor
    layer_name: str  # name of the layer that the tensor belongs to
    param_num: Optional[int] = None  # parameter index in model.parameters()


class MemoryProfiler(BaseProfiler):
    def __init__(self, layer_names: List[str] = None):
        """Memory profiler for tracking layer-wise memory usage

        Args:
            layer_names: List of layer names to profile, sorted in forward pass order
        """
        super().__init__(layer_names)

        self.reset_metrics()

    def reset_metrics(self):
        """Reset all memory tracking metrics"""
        self._init_layer_memory_metrics()
        self._init_total_memory_metrics()
        self.max_reserved_gib = []

    def log_activation_memory_info(self, saved_tensor_mem_layer: list[float]):
        """Log activation memory usage for each layer

        Args:
            saved_tensor_mem_layer: List of memory usage values per layer in MB

        Raises:
            ValueError: If length of memory usage list doesn't match number of layers
        """
        if len(saved_tensor_mem_layer) != len(self.layer_names):
            raise ValueError("Number of layers and memory usage list should match")

        for layer_mem, ln in zip(saved_tensor_mem_layer, self.layer_names):
            self.activation_memory_usage[ln].append(layer_mem)

        self.total_activation_mem_size.append(sum(saved_tensor_mem_layer))

    def _get_tensor_size_mb(self, tensor: torch.Tensor) -> float:
        """Get tensor size in megabytes

        Args:
            tensor: Input tensor

        Returns:
            Size in MB as float
        """
        if isinstance(tensor, torch.distributed.tensor.DTensor):
            bytes = tensor.to_local().untyped_storage().nbytes()
        else:
            bytes = tensor.untyped_storage().nbytes()
        return bytes / (1024 * 1024)

    def _get_tensor_storage_id(self, tensor: torch.Tensor) -> int:
        """Get unique storage ID for tensor

        Args:
            tensor: Input tensor

        Returns:
            Storage ID as integer
        """
        if isinstance(tensor, torch.distributed.tensor.DTensor):
            return id(tensor.to_local().untyped_storage())
        return id(tensor.untyped_storage())

    def log_weight_grad_optimizer_memory_info(self, model, optimizers, device):
        """Log weight, gradient and optimizer memory usage per layer

        Args:
            model: PyTorch model
            optimizers: Model optimizer
            device: Target device
        """

        # get weight and gradient memory usage
        tensor_info, untracked_tensor_info = self._get_log_weight_grad_memory_info(model, device)

        # map parameters to layers
        param_num_layer_name = self._map_parameters_to_layers(
            model, tensor_info, untracked_tensor_info
        )

        # log optimizer memory usage
        self._log_optimizer_memory(optimizers, param_num_layer_name)

    def _get_log_weight_grad_memory_info(
        self, 
        model: torch.nn.Module,
        device: torch.device
    ) -> Dict[str, TensorMemoryInfo]:
        """Calculate weight and gradient memory usage per layer
        
        Args:
            model: PyTorch model
            device: Target device
            
        Returns:
            Dictionaries of tensor memory info for tracked and untracked tensors
        """
        tensor_info = {}
        untracked_tensor_info = {}
        total_weight_size = 0
        total_grad_size = 0

        for layer_name, layer in model.named_modules():
            if layer_name not in self.layer_names:
                for param in layer.parameters():
                    assert param.device == device
                    storage_id = self._get_tensor_storage_id(param)
                    untracked_tensor_info[storage_id] = TensorMemoryInfo(
                        storage_id, layer_name
                    )                

            weight_size = 0
            grad_size = 0
            
            for param in layer.parameters():
                if param.device != device:
                    raise ValueError(f"Parameter of layer {layer_name} not on target device {device}, found on {param.device}")
                storage_id = self._get_tensor_storage_id(param)
                tensor_info[storage_id] = TensorMemoryInfo(storage_id, layer_name)
                
                weight_size += self._get_tensor_size_mb(param)
                if param.grad is not None:
                    grad_size += self._get_tensor_size_mb(param.grad)

            self.weight_memory_usage[layer_name].append(weight_size)
            self.grad_memory_usage[layer_name].append(grad_size)
            total_weight_size += weight_size
            total_grad_size += grad_size


        self.total_weight_mem_size.append(total_weight_size)
        self.total_grad_mem_size.append(total_grad_size)
        
        return tensor_info, untracked_tensor_info

    def _map_parameters_to_layers(
        self,
        model: torch.nn.Module,
        tensor_info: Dict[str, TensorMemoryInfo],
        untracked_tensor_info: Dict[str, TensorMemoryInfo],
    ) -> Dict[int, TensorMemoryInfo]:
        """Map parameter indices to layer information

        Args:
            model: PyTorch model
            tensor_info: Dictionary of tensor storage info
            untracked_tensor_info: Dictionary of untracked tensor storage info

        Returns:
            Dictionary mapping parameter indices to layer info
        """
        param_mapping = {}
        for param_num, param in enumerate(model.parameters()):
            storage_id = self._get_tensor_storage_id(param)
            if storage_id in tensor_info:
                info = tensor_info[storage_id]
                info.param_num = param_num
                param_mapping[param_num] = info
            else:
                # it must be in untracked_tensor_info
                if storage_id not in untracked_tensor_info:
                    raise ValueError(
                        f"Layer not found for the parameter with storage id {storage_id}"
                    )

        return param_mapping

    def _log_optimizer_memory(
        self, optimizer: Optimizer, param_2_layer_mapping: Dict[int, TensorMemoryInfo]
    ):
        """Calculate optimizer state memory usage per layer

        Args:
            optimizer: Model optimizer
            param_2_layer_mapping: Parameter number (index in the model.parameters()) to layer mapping
        """

        optimizer_mem_layer = {layer: 0 for layer in self.layer_names}
        total_optimizer_mem = 0

        state_dict = optimizer.state_dict()["state"]
        for param_idx, values in state_dict.items():

            if param_idx not in param_2_layer_mapping:
                raise ValueError(
                    f"Layer not found for the parameter with index {param_idx} in the param_2_layer_mapping"
                )

            layer_name = param_2_layer_mapping[param_idx].layer_name
            for tensor in values:
                optimizer_mem_layer[layer_name] += self._get_tensor_size_mb(tensor)
                total_optimizer_mem += self._get_tensor_size_mb(tensor)

        for layer, mem in optimizer_mem_layer.items():
            self.optimizer_memory_usage[layer].append(mem)
        self.total_optimizer_mem_size.append(total_optimizer_mem)

    def get_memory_usage(self):
        """Get all memory usage metrics

        Returns:
            Dictionary containing all memory metrics
        """
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

    def log_max_reserved_gib(self, max_reserved_gib):
        self.max_reserved_gib.append(max_reserved_gib)

    def _init_layer_memory_metrics(self):
        """Initialize per-layer memory tracking dictionaries"""
        self.activation_memory_usage = {}
        self.weight_memory_usage = {}
        self.grad_memory_usage = {}
        self.optimizer_memory_usage = {}

        for ln in self.layer_names:
            self.activation_memory_usage[ln] = []
            self.weight_memory_usage[ln] = []
            self.grad_memory_usage[ln] = []
            self.optimizer_memory_usage[ln] = []

    def _init_total_memory_metrics(self):
        """Initialize total memory tracking lists"""
        self.total_activation_mem_size = []
        self.total_weight_mem_size = []
        self.total_grad_mem_size = []
        self.total_optimizer_mem_size = []
