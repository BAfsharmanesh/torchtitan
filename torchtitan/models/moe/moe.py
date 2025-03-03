import torch
import torch.nn as nn
from mixture_of_experts import HeirarchicalMoE
from dataclasses import dataclass, field
from torchtitan.models.utils import weights_init




@dataclass
class ModelArgs:
    hidden_size: int = 1024
    n_layers: int = 16
    heads: int = 16
    experts: int = 16
    num_classes: int = 10
    dim: int = field(init=False)
    num_experts : tuple = field(init=False)
    
    def __post_init__(self):
        self.dim = self.hidden_size//4
        self.num_experts = (self.heads, self.experts)
    
#### MOE MODEL ####

class MultiLayerHierarchicalMoE(torch.nn.Module):
    def __init__(self, model_args: ModelArgs):
        """ (dim, num_experts, num_layers, num_classes)
        Multi-layer Hierarchical Mixture of Experts (MoE) model.

        model_args:
            dim (int): Dimension of the input features.
            num_experts (tuple): Number of experts in each hierarchy level for each layer.
            layers (int): Number of Hierarchical MoE layers.
        """
        super(MultiLayerHierarchicalMoE, self).__init__()

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = HeirarchicalMoE(dim=model_args.dim, num_experts=model_args.num_experts)       
        
        # self.is_first = True
        # self.is_last = True
        self.classifier = nn.Linear(model_args.dim, model_args.num_classes)
        
    def init_weights(self):
        """Initializes the weights of the model."""
        for m in self.modules():
            m.apply(weights_init)

    def forward(self, x, passed_aux_loss=0):
        
        # forward pass through the layers
        aux_loss_total = 0.0
            
        for layer in self.layers.values():
            if layer is not None:
                x, aux_loss = layer(x)
                aux_loss_total += aux_loss

        if self.classifier:
            logits = self.classifier(x)
            return logits, aux_loss_total+passed_aux_loss
        else:
            return x, aux_loss_total+passed_aux_loss
        
    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "MultiLayerHierarchicalMoE":
        """
        Initialize a MultiLayerHierarchicalMoE model from a ModelArgs object.

        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            MultiLayerHierarchicalMoE: MultiLayerHierarchicalMoE model.

        """
        return cls(model_args)        
#### MOE MODEL ####