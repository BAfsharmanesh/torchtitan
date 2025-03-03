from torchtitan.models.wideresnet.model import WideResNet, ModelArgs


    
wideresnet_configs = {
    "250M": ModelArgs(n_tot_layers=50, num_filters=160, width_factor=2),
    "1B": ModelArgs(n_tot_layers=50, num_filters=320, width_factor=2),
    "2B": ModelArgs(n_tot_layers=50, num_filters=448, width_factor=2),
    "4B": ModelArgs(n_tot_layers=50, num_filters=640, width_factor=2),
    "6.8B": ModelArgs(n_tot_layers=50, num_filters=320, width_factor=16),
    "13B": ModelArgs(n_tot_layers=101, num_filters=320, width_factor=16),
}





