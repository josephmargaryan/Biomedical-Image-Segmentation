import numpy as np
import torch
import torch.nn as nn

# Provided corpus and sample
corpus = "Hi my name is Joseph and i am studying about transformers from the litterature"
tokens = corpus.split()
unique_tokens = set(tokens)
indices = np.arange(0, len(unique_tokens))
dictionary = dict(zip(indices, unique_tokens))

sample = "the name is Joseph"
sample_tokens = sample.split()
reverse_dictionary = {token: idx for idx, token in dictionary.items()}
mapped_indices = [reverse_dictionary[token] for token in sample_tokens if token in reverse_dictionary]

input_tensor = torch.tensor(np.array(mapped_indices), dtype=torch.long)

# Embedding
embed_dim = 16  # Embedding dimension set to 16
embed_func = nn.Embedding(len(dictionary), embed_dim)

embeddings = embed_func(input_tensor)

# Positional Encoding
def positional_encoding(seq_len, d, n=10000):
    """
    P(k, 2i) = sin(k / n^(2i/d))
    P(k, 2i+1) = cos(k / n^(2i/d))
    """
    P = torch.zeros(seq_len, d)
    for k in range(seq_len):
        for i in range(0, d, 2):
            denominator = n ** (2 * i / d)
            P[k, i] = np.sin(k / denominator)
            if i + 1 < d:
                P[k, i + 1] = np.cos(k / denominator)
    return P

seq_len = embeddings.shape[0]
pos_encoding = positional_encoding(seq_len, embed_dim)

# Adding positional encoding to embeddings
input_with_pos_encoding = embeddings + pos_encoding

# Transformer Encoder Layer
class SimpleTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(SimpleTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x

# Instantiate the encoder layer
embed_dim = 16
num_heads = 2
ff_dim = 32

encoder_layer = SimpleTransformerEncoderLayer(embed_dim, num_heads, ff_dim)

# Add a batch dimension for the input
input_with_pos_encoding = input_with_pos_encoding.unsqueeze(1)

# Forward pass through the encoder layer
output = encoder_layer(input_with_pos_encoding)


