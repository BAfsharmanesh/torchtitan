from typing import Dict, List, Optional, Any, Set, Iterable
import torch
import contextlib
import weakref

class SavedActivationContext:
    """Context manager for tracking activation memory during model execution."""
    
    def __init__(self, layer_names: Optional[List[str]] = None, ignored_tensors: Optional[Iterable[torch.Tensor]] = None) -> None:
        self._ignored_data_ptrs = (
            set() if ignored_tensors is None
            else {
                id(t.to_local().untyped_storage())
                if isinstance(t, torch.distributed.tensor.DTensor)
                else id(t.untyped_storage())
                for t in ignored_tensors
            }
        )
        
        self.layer_names = layer_names
        self.saved_tensor_dict = torch.utils.weak.WeakTensorKeyDictionary()
        self.saved_tensor_list = WeakTensorList()
        self.layer_pos = [0]  # Initialize with first position

        def pack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            data_ptr = (
                id(saved_tensor.to_local().untyped_storage())
                if isinstance(saved_tensor, torch.distributed.tensor.DTensor)
                else id(saved_tensor.untyped_storage())
            )
            if data_ptr not in self._ignored_data_ptrs:
                self.saved_tensor_dict[
                    saved_tensor.to_local() 
                    if isinstance(saved_tensor, torch.distributed.tensor.DTensor)
                    else saved_tensor
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

    def take_layer_snapshot(self, layer_name: str) -> None:
        """Take a snapshot of tensor list length at current layer."""
        if self.layer_names and layer_name in self.layer_names:
            self.layer_pos.append(len(self.saved_tensor_list))

    def __enter__(self) -> "SavedActivationContext":
        self._saved_tensors_hook.__enter__()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self._saved_tensors_hook.__exit__(*args, **kwargs)

    @property
    def saved_tensor_mem_layer(self) -> List[float]:
        """Get memory usage per layer in MB."""
        if len(self.layer_pos) <= 1:
            return []
            
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
                    
            total_bytes_list.append(total_bytes / (1024 * 1024))  # Convert to MB
            
        return total_bytes_list

class WeakTensorList:
    """List-like container that holds weak references to tensors."""
    
    def __init__(self):
        self._refs = []

    def append(self, tensor):
        # Add a weak reference to the tensor
        self._refs.append(weakref.ref(tensor))

    def __getitem__(self, index):
        # Retrieve the tensor, if it's still alive
        tensor_ref = self._refs[index]()
        return tensor_ref

    def __len__(self):
        return len(self._refs)

    def cleanup(self):
        # Clean up any None references from the list
        self._refs = [ref for ref in self._refs if ref() is not None]

@contextlib.contextmanager
def measure_activation_shape(model: torch.nn.Module, layer_names: Optional[List[str]] = None):
    """Context manager for measuring activation shapes during model execution.
    
    Args:
        model: PyTorch model to profile
        layer_names: Optional list of layer names to track
        
    Yields:
        SavedActivationContext instance
    """
    context = SavedActivationContext(layer_names)
    context.register_hooks(model)
    try:
        yield context
    finally:
        context.remove_hooks() 