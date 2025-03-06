import weakref
from typing import Any, Iterable, Optional

import torch


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
        
        
def measure_activation_shape(model, layers_to_monitor, dummy_input):

    # copy and move dummy_input to meta device
    dummy_input = dummy_input.clone().to("meta")

    # Measure the activation shape
    # Register hooks
    activations = {}
    activations_size = {}
    input_shapes = {}

    def get_hook(module_name):
        def hook_fn(module, input, output):
            if output is not None:
                # if isinstance(output, tuple) then keep both shapes
                if isinstance(output, tuple):
                    activations[module_name] = output[0].shape
                    activations_size[module_name] = (
                        output[0].nelement() * output[0].element_size() / 1024 / 1024
                    )
                else:
                    activations[module_name] = output.shape
                    activations_size[module_name] = (
                        output.nelement() * output.element_size() / 1024 / 1024
                    )
            if input is not None:
                if isinstance(input, tuple):
                    input_shapes[module_name] = (i.shape for i in input)
                else:
                    input_shapes[module_name] = input.shape

        return hook_fn

    hooks = []
    for name, module in model.named_modules():
        if name in layers_to_monitor:
            hook = module.register_forward_hook(get_hook(name))
            hooks.append(hook)

    # Run the model on meta device
    device = torch.device("meta")
    # dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        with torch.device("meta"):
            model = model.to(device)
            model.eval()
            model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return list(activations.values()), list(activations_size.values()), input_shapes