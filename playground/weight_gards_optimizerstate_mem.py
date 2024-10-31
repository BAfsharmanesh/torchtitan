import weakref
from typing import Any, Iterable, Optional, Union

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.lin_out = nn.Linear(self.d_model*4*self.d_model, 1, **factory_kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.lin_out(x)
        return torch.sigmoid(x)


if __name__ == "__main__":
    batch_size, seq_len, d_model = 16, 4096, 1024
    dtype = torch.bfloat16
    inputs = torch.randn(
        batch_size,
        seq_len,
        d_model,
        device=device,
        requires_grad=True,
        dtype=dtype,
    )
    
    target = torch.randn(
        batch_size,
        1,
        device=device,
        requires_grad=True,
        dtype=dtype,
    )
    


    act_fn_dict = {"ReLU": nn.ReLU(), "GELU": nn.GELU()}
    # Append outputs to a list to keep tensors alive
    outputs = []
    mem_bytes = []

    # each layer activation function memory comparison
    mlp_model = MLP(
        d_model=d_model,
        act_fn=act_fn_dict["ReLU"],
        device=device,
        dtype=dtype,
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(mlp_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    
    
    output = mlp_model(inputs)
    
            
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    # optimizer.zero_grad()
    
    
    # print weight size for each layer
    name_list = ["layer1", "layer2", "layer3", "layer4", "lin_out"]
    for i, (name, op) in enumerate(mlp_model.named_modules()):
        if name in name_list:
            print(name, 'weights', sum([t.untyped_storage().nbytes() for t in op.parameters()]))  
            print(name, 'grads', sum([t.grad.untyped_storage().nbytes() for t in op.parameters()]))    
        
    state = optimizer.state_dict()['state']
    for k in state.keys():
        print(k, "state", [t.untyped_storage().nbytes() for t in state[k].values()])
    print(len([i for i in mlp_model.parameters()]))
