import torch
import time



def register_timing_hooks(
    model, timings, memory_usage, hook_layers, func=None, hooks=None
):

    def start_time(layer_name, pass_type, func=None):
        def hook(module, input):
            torch.cuda.synchronize()  # Ensure all CUDA operations are finished
            timings.setdefault(f"{layer_name}_{pass_type}_start", []).append(
                time.time()
            )
            torch.cuda.reset_peak_memory_stats()
            if func is not None:
                func()

        return hook

    def end_time(layer_name, pass_type, func=None):
        def hook(module, input, output):
            torch.cuda.synchronize()  # Ensure all CUDA operations are finished
            timings.setdefault(f"{layer_name}_{pass_type}_end", []).append(time.time())

            memory_usage.setdefault(
                f"{layer_name}_{pass_type}_start_reserved", []
            ).append(torch.cuda.max_memory_reserved())

            memory_usage.setdefault(
                f"{layer_name}_{pass_type}_start_allocated", []
            ).append(torch.cuda.max_memory_allocated())

            if func is not None:
                func()

        return hook

    # Iterate over each layer and register hooks
    # Only apply hooks to container-like layers, not leaf layers
    # hook_layers = [f"layers.{i}" for i in range(10)]
    # hook_layers = hook_layers + ["norm", "output"]

    for name, layer in model.named_modules():
        if name in hook_layers:
            if name in hooks:
                for layer_hooks in hooks[name]:
                    layer_hooks.remove()
                # delete key name from hooks
                del hooks[name]
            # print("Registering hooks for layer", name)
            h1 = layer.register_forward_pre_hook(start_time(name, "forward"))
            h2 = layer.register_forward_hook(end_time(name, "forward", func))
            h3 = layer.register_full_backward_pre_hook(start_time(name, "backward"))
            h4 = layer.register_full_backward_hook(end_time(name, "backward"))
            hooks[name] = [h1, h2, h3, h4]
        # elif name in ["tok_embeddings"]:
        #     if name in hooks:
        #         for layer_hooks in hooks[name]:
        #             layer_hooks.remove()
        #         del hooks[name]
        #     h1 = layer.register_forward_pre_hook(start_time(name, "forward"))
        #     h2 = layer.register_forward_hook(end_time(name, "forward", func))
        #     hooks[name] = [h1, h2]