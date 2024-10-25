import torch
import torch.nn as nn
from typing import Optional, Union, Any, Iterable

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    """
    Basic MLP (multi-layer perceptron) layer with optional Dropout.
    """

    def __init__(
        self,
        d_model: int,
        act_fn: nn.Module,
        dropout_prob: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.act_fn = act_fn
        self.dropout_prob = dropout_prob
        factory_kwargs = {"device": device, "dtype": dtype}

        self.lin_0 = nn.Linear(self.d_model, 4 * self.d_model, **factory_kwargs)
        self.lin_1 = nn.Linear(4 * self.d_model, self.d_model, **factory_kwargs)
        self.dropout = nn.Dropout(self.dropout_prob) if self.dropout_prob else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.lin_0(inputs)
        x = self.act_fn(x)
        x = self.lin_1(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
    
    
class AllocatedMemContext:
    def __init__(self) -> None:
        # Ensure CUDA libraries are loaded:
        torch.cuda.current_blas_handle()

        self.before: dict[str, int] = {}
        self.after: dict[str, int] = {}
        self.delta: dict[str, int] = {}

    def _get_mem_dict(self) -> dict[str, int]:
        # Only need `allocated_bytes.all`-prefixed keys here
        key_prefix = "allocated_bytes.all."
        return {
            k.replace(key_prefix, ""): v
            for k, v in torch.cuda.memory_stats().items()
            if key_prefix in k
        }

    def __enter__(self) -> "AllocatedMemContext":
        self.before = self._get_mem_dict()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.after = self._get_mem_dict()
        self.delta = {k: v - self.before[k] for k, v in self.after.items()}
        

class SavedTensorContext:
    def __init__(
        self,
        ignored_tensors: Optional[Iterable[torch.Tensor]] = None,
    ) -> None:
        self._ignored_data_ptrs = (
            set()
            if ignored_tensors is None
            else {t.untyped_storage().data_ptr() for t in ignored_tensors}
        )

        self.saved_tensor_dict = torch.utils.weak.WeakTensorKeyDictionary()

        def pack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            data_ptr = saved_tensor.untyped_storage().data_ptr()
            if data_ptr not in self._ignored_data_ptrs:
                self.saved_tensor_dict[saved_tensor] = data_ptr
            return saved_tensor

        def unpack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            return saved_tensor

        self._saved_tensors_hook = torch.autograd.graph.saved_tensors_hooks(
            pack_hook, unpack_hook
        )

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
            data_ptr = t.untyped_storage().data_ptr()
            if data_ptr not in accounted_for:
                total_bytes += t.untyped_storage().nbytes()
                accounted_for.add(data_ptr)
        return total_bytes


if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 4096, 1024
    dtype = torch.bfloat16
    inputs = torch.randn(
        batch_size,
        seq_len,
        d_model,
        device=device,
        requires_grad=True,
        dtype=dtype,
    )

    act_fn_dict = {"ReLU": nn.ReLU(), "GELU": nn.GELU()}
    # Append outputs to a list to keep tensors alive
    outputs = []
    mem_bytes = []

    for name, act_fn in act_fn_dict.items():
        mlp = MLP(
            d_model=d_model,
            act_fn=act_fn,
            device=device,
            dtype=dtype,
        )
        with AllocatedMemContext() as mem, SavedTensorContext(
            ignored_tensors=mlp.parameters()
        ) as saved:
            out = mlp(inputs)
            outputs.append(out)
        assert mem.delta["current"] == saved.saved_tensor_mem
        print(f"{name} bytes: {saved.saved_tensor_mem}")
        mem_bytes.append(saved.saved_tensor_mem)

    print(f"ReLU/GeLU act mem ratio: {mem_bytes[0]/mem_bytes[1]}")                    