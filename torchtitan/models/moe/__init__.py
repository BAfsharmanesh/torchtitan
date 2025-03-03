from torchtitan.models.moe.moe import MultiLayerHierarchicalMoE, ModelArgs


    
moe_configs = {
    "380M": ModelArgs(hidden_size=768, n_layers=8, heads=16, experts=8),
    "1.3B": ModelArgs(hidden_size=768, n_layers=16, heads=16, experts=16),
    "2.4B": ModelArgs(hidden_size=1024, n_layers=16, heads=16, experts=16),
    "10B": ModelArgs(hidden_size=1536, n_layers=16, heads=16, experts=32),
    "27B": ModelArgs(hidden_size=1536, n_layers=16, heads=32, experts=48),
}