import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

# Define model parameters
num_layers = 4   # Custom number of transformer layers
hidden_size = 256
num_heads = 8
seq_length = 10
vocab_size = 30522  # Standard BERT vocab size for simplicity

# Create a random batch of input data
batch_size = 1
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
