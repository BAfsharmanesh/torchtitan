import weakref
from typing import Any, Iterable, Optional, Union

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model parameters
num_layers = 16   # Custom number of transformer layers
hidden_size = 256
num_heads = 8
seq_length = 10
vocab_size = 30522  # Standard BERT vocab size for simplicity

# Create a random batch of input data
batch_size = 512
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

# Configure the BERT model with custom layers
config = BertConfig(
    hidden_size=hidden_size,
    num_attention_heads=num_heads,
    num_hidden_layers=num_layers,
    intermediate_size=hidden_size * 4,
    vocab_size=vocab_size
)

# Initialize the model and randomly initialize weights
model = BertModel(config)
model.apply(lambda m: torch.nn.init.normal_(m.weight) if hasattr(m, 'weight') else None)


# Forward pass with random input batch
outputs = model(input_ids)
print("Output shape:", outputs.last_hidden_state.shape)


class WeakTensorList:
    def __init__(self):
        self._refs = []

    def append(self, tensor):
        # Add a weak reference to the tensor
        self._refs.append(weakref.ref(tensor))

    def __getitem__(self, index):
        # Retrieve the tensor, if it's still alive
        tensor_ref = self._refs[index]()
        if tensor_ref is None:
            print(f"Tensor at index {index} has been garbage collected.")
        return tensor_ref

    def __len__(self):
        return len(self._refs)

    def cleanup(self):
        # Clean up any None references from the list
        self._refs = [ref for ref in self._refs if ref() is not None]


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
        self.lin_2 = nn.Linear(self.d_model, self.d_model, **factory_kwargs)
        self.dropout = nn.Dropout(self.dropout_prob) if self.dropout_prob else None

        self.layer1 = nn.Sequential(self.lin_0, self.act_fn)
        self.layer2 = nn.Sequential(self.lin_1, self.act_fn)
        self.layer3 = nn.Sequential(self.lin_2, self.act_fn)
        self.layer4 = nn.Sequential(self.lin_0, self.lin_1)
        self.lin_out = nn.Linear(self.d_model * 4 * self.d_model, 1, **factory_kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1).contiguous()
        x = self.lin_out(x)
        return torch.sigmoid(x)


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
            else {id(t.untyped_storage()) for t in ignored_tensors}
        )

        self.saved_tensor_dict = torch.utils.weak.WeakTensorKeyDictionary()
        self.saved_tensor_list = WeakTensorList()
        self.layer_pos = [
            0,
        ]

        def pack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            data_ptr = id(saved_tensor.untyped_storage())
            if data_ptr not in self._ignored_data_ptrs:
                self.saved_tensor_dict[saved_tensor] = data_ptr
                self.saved_tensor_list.append(saved_tensor)
                print("packed:", saved_tensor.untyped_storage().nbytes() / 1024 / 1024)
            return saved_tensor

        def unpack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            print("unpacked:", saved_tensor.untyped_storage().nbytes() / 1024 / 1024)
            return saved_tensor

        self._saved_tensors_hook = torch.autograd.graph.saved_tensors_hooks(
            pack_hook, unpack_hook
        )

    def take_layer_pos(self):
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
                print("saved:", t.untyped_storage().nbytes() / 1024 / 1024)
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
                data_ptr = id(t.untyped_storage())
                if data_ptr not in accounted_for:
                    total_bytes += t.untyped_storage().nbytes()
                    accounted_for.add(data_ptr)
            total_bytes_list.append(total_bytes)
        return total_bytes_list


import pickle

if __name__ == "__main__":
    # how much memory allocation/free ops to record in memory snapshots
    MEMORY_SNAPSHOT_MAX_ENTRIES = 100000
    # batch_size, seq_len, d_model = 32, 4096, 1024
    # dtype = torch.bfloat16
    # inputs = torch.randn(
    #     batch_size,
    #     seq_len,
    #     d_model,
    #     device=device,
    #     requires_grad=True,
    #     dtype=dtype,
    # )

    # act_fn_dict = {"ReLU": nn.ReLU(), "GELU": nn.GELU()}
    # # Append outputs to a list to keep tensors alive
    # outputs = []
    # mem_bytes = []

    torch.cuda.memory._record_memory_history(max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES)
    # each layer activation function memory comparison
    # mlp_model = MLP(
    #     d_model=d_model,
    #     act_fn=act_fn_dict["ReLU"],
    #     device=device,
    #     dtype=dtype,
    # )
    
    model.to(device)
    input_ids = input_ids.to(device)
    
    with AllocatedMemContext() as mem, SavedTensorContext(
        ignored_tensors=model.parameters()
    ) as saved:
        # out = mlp_model(inputs)
        outputs = model(input_ids)

        print(f"total bytes: {saved.saved_tensor_mem}")

        # del inputs, out

    with open("memory_snapshot.pickle", "wb") as output:
        pickle.dump(torch.cuda.memory._snapshot(), output)

    print(f"total bytes: {saved.saved_tensor_mem}")
