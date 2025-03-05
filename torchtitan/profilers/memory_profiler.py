from typing import Dict, List, Optional, Any
import torch
from .base_profiler import BaseProfiler

class MemoryProfiler(BaseProfiler):
    """Profiles memory usage of model layers, including activations, weights, gradients and optimizer states."""
    
    def __init__(self, layer_names: Optional[List[str]] = None):
        """Initialize the memory profiler.
        
        Args:
            layer_names: List of layer names to profile. Must be sorted in forward pass order.
        """
        super().__init__(layer_names)
        self.reset()

    def reset(self) -> None:
        """Reset all memory usage measurements."""
        self.activation_memory_usage = {ln: [] for ln in self.layer_names}
        self.weight_memory_usage = {ln: [] for ln in self.layer_names}
        self.grad_memory_usage = {ln: [] for ln in self.layer_names}
        self.optimizer_memory_usage = {ln: [] for ln in self.layer_names}
        
        # Total memory tracking
        self.total_activation_mem_size: List[float] = []
        self.total_weight_mem_size: List[float] = []
        self.total_grad_mem_size: List[float] = []
        self.total_optimizer_mem_size: List[float] = []
        self.max_reserved_gib: List[float] = []


    def log_activation_memory_info(self, saved_tensor_mem_layer: List[float]) -> None:
        """Log activation memory usage for each layer.
        
        Args:
            saved_tensor_mem_layer: List of memory usage in MB for each layer's activations
        """
        assert len(saved_tensor_mem_layer) == len(self.layer_names), \
            "Number of layers and memory usage list should match"
            
        for layer_mem, ln in zip(saved_tensor_mem_layer, self.layer_names):
            self.activation_memory_usage[ln].append(layer_mem)
        self.total_activation_mem_size.append(sum(saved_tensor_mem_layer))

    def log_weight_grad_optimizer_memory_info(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> None:
        """Log memory usage for weights, gradients and optimizer states.
        
        Args:
            model: PyTorch model being profiled
            optimizer: Optimizer being used
            device: Device where model/optimizer reside
        """
        # Track parameter storage IDs to layer mapping
        id_layer_param_num: Dict[int, Dict[str, Any]] = {}
        id_param_num_untracked: Dict[int, Dict[str, Any]] = {}
        
        total_weight_size = 0
        total_grad_size = 0

        # Collect memory usage per layer
        for name, layer in model.named_modules():
            if name in self.layer_names:
                # Get memory usage for parameters
                weight_size = self._get_storage_size(layer.parameters()) / (1024 * 1024)
                grad_size = self._get_grad_size(layer.parameters()) / (1024 * 1024)
                
                self.weight_memory_usage[name].append(weight_size)
                self.grad_memory_usage[name].append(grad_size)
                total_weight_size += weight_size
                total_grad_size += grad_size
                
                # Track parameter storage IDs
                for param in layer.parameters():
                    storage_id = self._get_storage_id(param)
                    id_layer_param_num[storage_id] = {"layer": name}
            else:
                for param in layer.parameters():
                    storage_id = self._get_storage_id(param)
                    id_param_num_untracked[storage_id] = {"layer": name}

        # Map parameters to their indices
        for param_idx, param in enumerate(model.parameters()):
            storage_id = self._get_storage_id(param)
            if storage_id in id_layer_param_num:
                id_layer_param_num[storage_id]["param_num"] = param_idx

        # Track optimizer state memory
        param_num_to_layer = {
            v["param_num"]: v["layer"] 
            for v in id_layer_param_num.values() 
            if "param_num" in v
        }
        
        optimizer_mem_layer = {layer: 0.0 for layer in self.layer_names}
        total_optimizer_mem = 0.0
        
        # Sum up optimizer state memory per layer
        state_dict = optimizer.state_dict()["state"]
        for param_idx, param_state in state_dict.items():
            if param_idx in param_num_to_layer:
                layer_name = param_num_to_layer[param_idx]
                state_mem = sum(
                    self._get_storage_size([t]) / (1024 * 1024)
                    for t in param_state.values()
                    if isinstance(t, torch.Tensor)
                )
                optimizer_mem_layer[layer_name] += state_mem
                total_optimizer_mem += state_mem

        # Record optimizer memory usage
        for layer, mem in optimizer_mem_layer.items():
            self.optimizer_memory_usage[layer].append(mem)
        self.total_optimizer_mem_size.append(total_optimizer_mem)
        
        # Record total memory usage
        self.total_weight_mem_size.append(total_weight_size)
        self.total_grad_mem_size.append(total_grad_size)

    def log_max_reserved_gib(self, max_reserved_gib: float) -> None:
        """Log maximum reserved GPU memory.
        
        Args:
            max_reserved_gib: Maximum reserved memory in GiB
        """
        self.max_reserved_gib.append(max_reserved_gib)

    def get_metrics(self) -> Dict[str, Any]:
        """Get the memory usage metrics for all layers."""
        metrics = {}
        for name in self.layer_names:
            metrics[name] = {
                "activation": sum(self.activation_memory_usage[name]) / len(self.activation_memory_usage[name]) if self.activation_memory_usage[name] else 0,
                "weights": sum(self.weight_memory_usage[name]) / len(self.weight_memory_usage[name]) if self.weight_memory_usage[name] else 0,
                "gradients": sum(self.grad_memory_usage[name]) / len(self.grad_memory_usage[name]) if self.grad_memory_usage[name] else 0,
                "optimizer": sum(self.optimizer_memory_usage[name]) / len(self.optimizer_memory_usage[name]) if self.optimizer_memory_usage[name] else 0
            }
        
        # Add totals
        metrics["total"] = {
            "activation": sum(self.total_activation_mem_size) / len(self.total_activation_mem_size) if self.total_activation_mem_size else 0,
            "weights": sum(self.total_weight_mem_size) / len(self.total_weight_mem_size) if self.total_weight_mem_size else 0,
            "gradients": sum(self.total_grad_mem_size) / len(self.total_grad_mem_size) if self.total_grad_mem_size else 0,
            "optimizer": sum(self.total_optimizer_mem_size) / len(self.total_optimizer_mem_size) if self.total_optimizer_mem_size else 0,
            "max_reserved_gib": max(self.max_reserved_gib) if self.max_reserved_gib else 0
        }
        
        return metrics

    @staticmethod
    def _get_storage_id(tensor: torch.Tensor) -> int:
        """Get unique storage ID for a tensor."""
        if isinstance(tensor, torch.distributed.tensor.DTensor):
            return id(tensor.to_local().untyped_storage())
        return id(tensor.untyped_storage())

    @staticmethod
    def _get_storage_size(tensors) -> int:
        """Get total storage size in bytes for a collection of tensors."""
        return sum(
            t.to_local().untyped_storage().nbytes()
            if isinstance(t, torch.distributed.tensor.DTensor)
            else t.untyped_storage().nbytes()
            for t in tensors
        )

    @staticmethod
    def _get_grad_size(parameters) -> int:
        """Get total gradient size in bytes for a collection of parameters."""
        return sum(
            (t.grad.to_local().untyped_storage().nbytes()
             if isinstance(t.grad, torch.distributed.tensor.DTensor)
             else t.grad.untyped_storage().nbytes())
            for t in parameters
            if t.grad is not None
        ) 