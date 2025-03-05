ACTIVATION_SAFTEY_FACTOR = {'wide_resnet':1, 'moe':2.2, 'llama2': 1.18}
TOTAL_SAFTEY_FACTOR = {'wide_resnet':1.27, 'moe':1.2, 'llama2': 1.27}

import torch

# get models layer names and activation size, weight size, etc.
def get_dummy_input(config, model_config):
    # Prepare a dummy input based on the specified input size
    if config.model.name == "llama2":
        dummy_input = torch.randint(
            0,
            model_config.vocab_size,
            (config.training.batch_size, config.training.seq_len),
            device="meta",
            dtype=torch.long,
        )
    elif config.model.name == "moe":
        batch_input_size = (
            config.training.batch_size,
            config.training.seq_len,
            model_config.dim,
        )
        dummy_input = torch.empty(
            *batch_input_size, device="meta", dtype=torch.float32
        )

    elif config.model.name == "wideresnet":
        batch_input_size = (
            config.training.batch_size,
            model_config.input_channels,
            model_config.input_size,
            model_config.input_size,
        )
        dummy_input = torch.empty(
            *batch_input_size, device="meta", dtype=torch.float32
        )
    else:
        raise ValueError(f"Unsupported model name: {config.model.name}")
    return dummy_input

def slice_layers_2_fit_gpu(act_weight_profiled, gpu_memory, tp_degree):
    # print(f"GPU Memory: {gpu_memory}")
    parameters_per_layer_bytes = act_weight_profiled["parameters_per_layer_bytes"]
    activation_parameters_bytes = act_weight_profiled["activation_parameters_bytes"]

    assert len(parameters_per_layer_bytes) == len(
        activation_parameters_bytes
    ), "Number of layers mismatch"
    number_of_layers = len(parameters_per_layer_bytes)

    # predict memory usage for each layer
    def _memory_usage_precidtions(weight, act, model_name):
        return TOTAL_SAFTEY_FACTOR[model_name]*(ACTIVATION_SAFTEY_FACTOR[model_name]*act + weight * 4)/tp_degree


    model_name = act_weight_profiled['model_name']
    model_name = "_".join(model_name.split('_')[:-1])
    # calculate size of each layer
    layer_size_tmp = []
    for i in range(number_of_layers):
        layer_size_tmp.append(
            _memory_usage_precidtions(
                parameters_per_layer_bytes[i], activation_parameters_bytes[i], model_name
            )/1024/1024/1024
        )
    # print(f"{layer_size_tmp=}")
    

    # start from layer 0, append layers until memory is full, then start a new slice
    model_slices = []
    current_slice = []
    current_slice_params = 0
    current_slice_activation = 0
    for i in range(number_of_layers):
        current_slice_params += parameters_per_layer_bytes[i]
        current_slice_activation += activation_parameters_bytes[i]
        if (
            _memory_usage_precidtions(current_slice_params, current_slice_activation, model_name)
            < gpu_memory
        ):
            current_slice.append(i)
        else:
            assert len(current_slice) != 0, "Some layers are too big"
            model_slices.append(current_slice)
            current_slice = []
            current_slice_params = parameters_per_layer_bytes[i]
            current_slice_activation = activation_parameters_bytes[i]
            current_slice.append(i)

    if current_slice:
        model_slices.append(current_slice)

    # check if all layers are included, and all slices are fit in the GPU memory
    assert (
        sum([len(slice) for slice in model_slices]) == number_of_layers
    ), "Some layers are missing"
    
    model_slice_memory = []
    for ii, slice in enumerate(model_slices):
        memory_slice_usage = _memory_usage_precidtions(
                sum([parameters_per_layer_bytes[i] for i in slice]),
                sum([activation_parameters_bytes[i] for i in slice]),
                model_name,
            )
        # print(f"Slice {ii} memory usage: {memory_slice_usage/1024/1024/1024:.2f}GiB")
        model_slice_memory.append(memory_slice_usage/1024/1024/1024)
        assert (
            memory_slice_usage
            <= gpu_memory
        ), f"The slice {i} is too big"

    return model_slices, model_slice_memory