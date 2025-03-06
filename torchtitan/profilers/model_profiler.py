import torch



class ModelProfiler:
    # get models in meta device, measure parameters and activations size per layer
    def __init__(self, model, layer_names=None):

        self.model = model.to("meta")
        self.layer_names = (
            layer_names if layer_names else [name for name, _ in model.named_modules()]
        )

    def get_total_parameters(self):
        # Calculate the total parameter size
        total_parameters_bytes = sum(
            p.element_size() * p.numel() for p in self.model.parameters()
        )
        return total_parameters_bytes

    def get_parameters_per_layer(self) -> list:
        parameters_per_layer_bytes = []
        for name, layer in self.model.named_modules():
            if name in self.layer_names:
                # Calculate the total parameter size for each layer
                total_params = sum(
                    p.element_size() * p.numel() for p in layer.parameters()
                )
                parameters_per_layer_bytes.append((name, total_params))
        return parameters_per_layer_bytes

    def get_activation_parameters_per_layer(self, dummy_input) -> list:

        # copy and move dummy_input to meta device
        dummy_input = dummy_input.clone().to("meta")

        activation_parameters_bytes = []

        def get_activation_hook(module_name):
            # Hook to capture activations
            def activation_hook(module, input, output):
                if output is not None:
                    # Calculate the activation size in bytes for the output
                    # if output is tuple, sum the size of each tensor
                    if isinstance(output, tuple):
                        activation_size = sum(
                            o.element_size() * o.numel() for o in output
                        )
                    else:
                        activation_size = output.element_size() * output.numel()
                    activation_parameters_bytes.append((module_name, activation_size))

            return activation_hook

        # Register hooks on specified layers
        hooks = []
        for module_name, layer in self.model.named_modules():
            if module_name in self.layer_names:
                # register hook for all submodules in the layer, if it has any, else register hook for the layer
                if len(list(layer.children())) > 0:
                    for name, submodule in layer.named_modules():
                        hook = submodule.register_forward_hook(
                            get_activation_hook(module_name)
                        )
                        hooks.append(hook)
                else:
                    hook = layer.register_forward_hook(get_activation_hook(module_name))
                    hooks.append(hook)
                # hook = layer.register_forward_hook(get_activation_hook(module_name))
                # hooks.append(hook)

        # Forward pass with dummy input to calculate activation sizes
        with torch.no_grad():
            self.model(dummy_input)

        # Remove hooks after calculation
        for hook in hooks:
            hook.remove()

        activation_parameters_bytes_dict = {}

        for i in activation_parameters_bytes:
            if i[0] not in activation_parameters_bytes_dict:
                activation_parameters_bytes_dict[i[0]] = i[1]
            else:
                activation_parameters_bytes_dict[i[0]] += i[1]

        return list(activation_parameters_bytes_dict.items())