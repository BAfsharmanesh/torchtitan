
## utils
import torch.nn.init as init
import torch.nn as nn

def weights_init(m):
    """
    Initializes the weights of the model.
    Uses Xavier initialization for Conv2d layers
    and sets biases to 0 for consistency.
    """
    if isinstance(m, nn.Conv2d):
        # Xavier initialization for weights
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            # Initialize biases to zero
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # Initialize BatchNorm layers
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # Xavier initialization for fully connected layers
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)